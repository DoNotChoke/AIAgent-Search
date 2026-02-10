import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from sentence_transformers import SentenceTransformer

from storage.config import config
from storage.embedding import Chunk, splitting, embedding
from storage.utils import stable_doc_id, ensure_bucket, upload_documents, build_s3uri, get_minio_client

import pandas as pd
from typing import List, Dict, Any

from feast import FeatureStore
from dotenv import load_dotenv


load_dotenv()

CHUNKS_BUCKET = config.bucket
TOPIC = config.kafka_document_topic
KAFKA_BOOTSTRAP_SERVERS = config.kafka_bootstrap_server
KAFKA_GROUP_ID = config.kafka_group_id


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kafka-consumer")


def upload_chunks_jsonl(minio_client, url: str, payload: str, retrieved_at: int):
    chunks = splitting(url, payload)
    doc_id = stable_doc_id(url)
    key = f"web/{doc_id}.jsonl"
    ensure_bucket(minio_client, CHUNKS_BUCKET)

    lines = []
    for c in chunks:
        lines.append(json.dumps({
            "chunk_id": c.chunk_id,
            "url": c.url,
            "idx": c.idx,
            "text": c.text,
        }, ensure_ascii=False))

    data = "\n".join(lines) + ('\n'if lines else "")
    upload_documents(
        client=minio_client,
        s3_uri=build_s3uri(CHUNKS_BUCKET, key),
        payload=data,
        metadata={"retrieved_at": retrieved_at},
    )
    return build_s3uri(CHUNKS_BUCKET, key), chunks


def embedding_and_store(store: FeatureStore, model:SentenceTransformer, url: str, s3_uri: str, chunks: List[Chunk], retrieved_at: int):
    vecs = embedding(model, chunks)
    doc_id = stable_doc_id(url)
    try:
        now = pd.Timestamp.now(tz="UTC")
        chunk_ids = [c.chunk_id for c in chunks]
        df = pd.DataFrame({
            config.url_col: [url] * len(chunks),
            config.doc_id_col: [doc_id] * len(chunks),
            config.chunk_id_col: chunk_ids,
            config.embedding_col: vecs,
            config.s3_uri_col: [s3_uri] * len(chunks),
            config.retrieved_at_col: [retrieved_at] * len(chunks),
            config.ts_col: [now] * len(chunks),
        })
        store.write_to_online_store(feature_view_name=config.feature_view_name, df=df)
    except Exception as e:
        print(f"Failed to upload {url}: {e}")

    print("==== DONE ====")

async def process_one(
        msg: Dict[str, Any],
        minio_client,
        store: FeatureStore,
        model: SentenceTransformer,
):
    meta = msg.get("metadata") or {}
    retrieved_at = meta.get("retrieved_at")
    url = meta.get("url") or ""

    payload = msg.get("payload") or ""

    if not url or not payload.strip():
        raise ValueError("Invalid message: missing url or payload")

    s3_uri, chunks = upload_chunks_jsonl(minio_client, url, payload, retrieved_at)
    logger.info("Uploaded chunks to MinIO: %s (chunks=%d)", s3_uri, len(chunks))

    embedding_and_store(store, model, url, s3_uri, chunks, retrieved_at)
    logger.info("Upserted embeddings to Feast(Milvus) for url=%s", url)


def parse_msg(value_bytes: bytes) -> Dict[str, Any]:
    return json.loads(value_bytes.decode("utf-8"))


async def main() -> None:
    logger.info("Starting consumer: topic=%s bootstrap=%s group_id=%s", TOPIC, KAFKA_BOOTSTRAP_SERVERS, KAFKA_GROUP_ID)

    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=parse_msg,
        max_poll_records=50
    )

    minio_client = get_minio_client(config.minio_endpoint, config.minio_access_key, config.minio_secret_key, config.minio_secure)
    store = FeatureStore(config.feast_docs_repo)
    model = SentenceTransformer(config.model_path)

    sem = asyncio.Semaphore(4)

    await consumer.start()
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=1000, max_records=50)

            tasks = []
            records_for_commit = []

            for tp, records in batch.items():
                for r in records:
                    records_for_commit.append((tp, r.offset))

                    async def run(record=r):
                        async with sem:
                            await process_one(
                                record.value,
                                minio_client,
                                store=store,
                                model=model,
                            )

                    tasks.append(asyncio.create_task(run()))

            if not tasks:
                continue

            results = await asyncio.gather(*tasks, return_exceptions=True)

            had_error = False
            for res in results:
                if isinstance(res, Exception):
                    had_error = True
                    logger.exception("Processing failed: %s", res)

            if not had_error:
                offsets = {}
                for tp, off in records_for_commit:
                    offsets[tp] = max(offsets.get(tp, -1), off)

                await consumer.commit({tp: off + 1 for tp, off in offsets.items()})
                logger.info("Committed offsets for %d records", len(records_for_commit))
            else:
                logger.warning("Skipped commit due to errors; messages will retry.")
    finally:
        await consumer.stop()

if __name__ == '__main__':
    asyncio.run(main())


