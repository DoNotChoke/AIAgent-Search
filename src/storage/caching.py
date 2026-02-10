import uuid
from dataclasses import dataclass

from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer
from feast import FeatureStore
from minio import Minio

import pandas as pd

from storage.utils import ensure_bucket, build_s3uri, upload_documents, download_text


@dataclass
class CacheHit:
    hit: bool
    answer: str = ""
    similarity: float = 0.0
    meta: Optional[Dict[str, Any]] = None


class SemanticCache:
    def __init__(
            self,
            store: FeatureStore,
            encoder: SentenceTransformer,
            minio_client: Minio,
            bucket: str = "semantic-cache",
            prefix: str = "qa",
            threshold: float = 0.80,
            top_k: int = 5,
            feature_view_name: str = "semantic_cache",
            vector_field: str = "embedding",
            distance_metric: str = "COSINE",
    ):
        self.store = store
        self.encoder = encoder
        self.minio = minio_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.threshold = threshold
        self.top_k = top_k
        self.feature_view_name = feature_view_name
        self.vector_field = vector_field
        self.distance_metric = distance_metric

        ensure_bucket(self.minio, self.bucket)
        self._feature_refs = [
            f"{self.feature_view_name}:{self.vector_field}",
            f"{self.feature_view_name}:cache_id",
            f"{self.feature_view_name}:question",
            f"{self.feature_view_name}:answer",
        ]

    def build_answer_s3_uri(self, cache_id: str) -> str:
        key = f"{self.prefix}/{cache_id}.txt"
        return build_s3uri(self.bucket, key)

    def upload_answer_to_minio(self, cache_id: str, question: str, answer: str) -> str:
        s3_uri = self.build_answer_s3_uri(cache_id)
        upload_documents(
            client=self.minio,
            s3_uri=s3_uri,
            payload=answer,
            content_type="text/plain",
        )
        return s3_uri

    def upload_to_cache(self, question: str, answer: str) -> str:
        question = (question or "").strip()
        answer = (answer or "").strip()

        cache_id = str(uuid.uuid4())

        # 1) upload answer to Minio
        answer_s3_uri = self.upload_answer_to_minio(cache_id, question, answer)

        # 2) embedding question
        vec = self.encoder.encode([question], normalize_embeddings=True).tolist()[0]
        now = pd.Timestamp.now(tz="UTC")


        df = pd.DataFrame(
            {
                "cache_id": [cache_id],
                "embedding": [vec],
                "question": [question],
                "answer": [answer_s3_uri],
                "event_timestamp": [now],
            }
        )

        self.store.write_to_online_store(
            feature_view_name=self.feature_view_name,
            df=df,
        )
        return cache_id

    def search_cache(self, question: str) -> CacheHit:
        question = (question or "").strip()
        if not question:
            return CacheHit(hit=False)

        qvec = self.encoder.encode([question], normalize_embeddings=True).tolist()[0]
        res = self.store.retrieve_online_documents_v2(
            features=self._feature_refs,
            query=qvec,
            top_k=self.top_k,
            distance_metric=self.distance_metric,
        ).to_df()

        if res is None or res.empty:
            return CacheHit(hit=False)

        distance_col = None
        for c in ("distance", "score", "similarity"):
            if c in res.columns:
                distance_col = c
                break

        best = res.iloc[0].to_dict()

        if distance_col is None:
            return CacheHit(hit=False, meta={"row": best})

        print(res)
        raw = best[distance_col]
        sim = raw

        print(sim)
        if sim < self.threshold:
            return CacheHit(hit=False, similarity=sim, meta={"best": best})

        answer_s3_uri = (best.get("answer") or "").strip()
        if not answer_s3_uri:
            return CacheHit(
                hit=False,
                similarity=sim,
                meta={"reason": "missing answer s3_uri in result row", "best": best},
            )

        try:
            full_answer = download_text(self.minio, answer_s3_uri)
            print(full_answer)
        except Exception as e:
            return CacheHit(
                hit=False,
                similarity=sim,
                meta={
                    "reason": "failed to download answer from minio",
                    "answer_s3_uri": answer_s3_uri,
                    "error": str(e),
                },
            )

        return CacheHit(
            hit=True,
            answer=full_answer,
            similarity=sim,
            meta={
                "cache_id": best.get("cache_id"),
                "question": best.get("question"),
                "answer_s3_uri": answer_s3_uri,
                "raw_metric": raw,
                "metric": self.distance_metric,
            },
        )