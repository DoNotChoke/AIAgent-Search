from dataclasses import dataclass

@dataclass
class Config:
    feast_docs_repo: str
    feast_cache_repo: str
    model_path: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool

    kafka_bootstrap_server: str
    kafka_max_request_size: int
    kafka_document_topic: str
    kafka_group_id: str

    bucket: str
    prefix: str
    recursive: bool = True

    feature_view_name_docs: str = "docs_embeddings"
    feature_view_name_cache: str = "semantic_cache"

    # FeatureView
    url_col: str = "url"
    doc_id_col: str = "doc_id"
    chunk_id_col: str = "chunk_id"
    embedding_col: str = "embedding"
    s3_uri_col: str = "s3_uri"
    retrieved_at_col: str = "retrieved_at"
    ts_col: str = "event_timestamp"


config = Config(
    feast_docs_repo="../infra/feature_store/docs",
    feast_cache_repo="../infra/feature_store/semantic",
    model_path="sentence-transformers/LaBSE",
    minio_endpoint="localhost:9000",
    minio_access_key="minioadmin",
    minio_secret_key="minioadmin",
    minio_secure=False,
    kafka_bootstrap_server="localhost:9092",
    kafka_max_request_size=10485760,
    kafka_document_topic="crawled_docs",
    kafka_group_id="crawled_docs_consumer",
    bucket="chunk-docs",
    prefix="web",
)
