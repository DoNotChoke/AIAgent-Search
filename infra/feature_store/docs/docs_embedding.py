from __future__ import annotations

from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Array, Float32, UnixTimestamp
from feast.value_type import ValueType


chunk = Entity(
    name="chunk",
    join_keys=["chunk_id"],
    value_type=ValueType.STRING,
    description="Unique chunk identifier, e.g., sha256(url)_{idx} or doc_id_{idx}.",
)

# (Optional) Entity: doc_id if you want doc-level joins later
doc = Entity(
    name="doc",
    join_keys=["doc_id"],
    value_type=ValueType.STRING,
    description="Unique document identifier, e.g., sha256(url).",
)

# -----------------------------
# Offline source: Parquet file
# -----------------------------
web_embeddings_source = FileSource(
    name="web_embeddings_source",
    path="../web_chunks_embeddings.parquet",
    timestamp_field="event_timestamp",
)

# -----------------------------
# FeatureView: web_chunk_embeddings
# -----------------------------
web_chunk_embeddings = FeatureView(
    name="docs_embeddings",
    entities=[chunk],
    schema=[
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE",
        ),

        Field(name="url", dtype=String),
        Field(name="doc_id", dtype=String),
        Field(name="s3_uri", dtype=String),
        Field(name="retrieved_at", dtype=String),

        Field(name="event_timestamp", dtype=UnixTimestamp),
    ],
    source=web_embeddings_source,
)
