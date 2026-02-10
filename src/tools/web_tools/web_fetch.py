import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, DefaultMarkdownGenerator, PruningContentFilter
from datetime import datetime, timezone
import asyncio

from aiokafka import AIOKafkaProducer

from typing import Optional, Dict, Any

from storage.config import config


TOPIC = config.kafka_document_topic

KAFKA_BOOTSTRAP_SERVERS = config.kafka_bootstrap_server
KAFKA_MAX_REQUEST_SIZE = config.kafka_max_request_size
KAFKA_PRODUCER: Optional[AIOKafkaProducer] = None


async def get_kafka_producer(bootstrap_servers, max_request_size) -> AIOKafkaProducer:
    global KAFKA_PRODUCER
    if KAFKA_PRODUCER is not None:
        return KAFKA_PRODUCER

    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",
        linger_ms=10,
        enable_idempotence=True,
        max_request_size=max_request_size,
        compression_type="gzip",
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
    )
    await producer.start()
    KAFKA_PRODUCER = producer
    return producer


async def publish_crawled_doc(
        bootstrap_servers: str,
        max_request_size: int,
        topic: str,
        key: str,
        value: Dict[str, Any]
) -> None:
    producer = await get_kafka_producer(bootstrap_servers, max_request_size)
    await producer.send_and_wait(topic, value=value, key=key)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


async def web_fetch(url: str) -> str:
    """
        Crawl a URL with crawl4ai and return fit_markdown.

        Args:
            url: Target URL
    """
    browser_config = BrowserConfig(
            headless=True,
            verbose=True,
        )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6, threshold_type="dynamic", min_word_threshold=0)
        ),
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )

    payload = (result.markdown.fit_markdown if result.markdown else "") or ""
    message = {
        "metadata": {
            "url": url,
            "retrieved_at": utc_now_iso(),
        },
        "payload": payload
    }
    asyncio.create_task(
        publish_crawled_doc(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            max_request_size=KAFKA_MAX_REQUEST_SIZE,
            topic=TOPIC,
            key=url,
            value=message,
        )
    )

    await publish_crawled_doc(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            max_request_size=KAFKA_MAX_REQUEST_SIZE,
            topic=TOPIC,
            key=url,
            value=message,
        )

    return payload