from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Array, Float32, UnixTimestamp
from feast.value_type import ValueType

cache_item = Entity(
    name="cache_item",
    join_keys=["cache_id"],
    value_type=ValueType.STRING,
    description="Unique cache id (uuid or stable hash of question)"
)

semantic_cache_source= FileSource(
    name="semantic_cache_source",
    path="../semantic_cache.parquet",
    timestamp_field="event_timestamp"
)

semantic_cache = FeatureView(
    name="semantic_cache",
    entities=[cache_item],
    schema=[
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE",
        ),
        Field(
            name="question",
            dtype=String
        ),
        Field(
            name="answer",
            dtype=String,
        ),
        Field(
            name="event_timestamp",
            dtype=UnixTimestamp
        )
    ],
    source=semantic_cache_source
)