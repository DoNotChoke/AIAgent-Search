import re
from feast import FeatureStore

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from storage.config import Config

import json
from typing import Dict, List, Set

from storage.utils import download_text, get_minio_client


def fetch_chunk_texts_from_jsonl(
    minio_client,
    s3_uri: str,
    chunk_ids: Set[str],
) -> Dict[str, str]:
    """
    Download JSONL from MinIO once, then extract text for the requested chunk_ids.
    Returns: {chunk_id: text}
    """
    if not chunk_ids:
        return {}

    text, _ = download_text(minio_client, s3_uri)

    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        cid = obj.get("chunk_id")
        if cid in chunk_ids:
            out[cid] = obj.get("text", "") or ""

            if len(out) == len(chunk_ids):
                break
    return out

@dataclass
class SearchHit:
    score: float
    chunk_id: str
    fields: Dict[str, Any]
    text: str


class NeuralSearcher:
    def __init__(
            self,
            cfg: Config,
            embedding_model: SentenceTransformer,
            normalize_embeddings: bool = True,
            return_fields: Optional[Sequence[str]] = None,
            distance_metric: str = "COSINE",
    ):
        self.minio_client = get_minio_client(
            cfg.minio_endpoint,
            cfg.minio_access_key,
            cfg.minio_secret_key,
            cfg.minio_secure
        )
        self.store = FeatureStore(cfg.feast_docs_repo)
        self.normalize_embeddings = normalize_embeddings
        if return_fields is None:
            return_fields = ["doc_id", "url", "s3_uri", "retrieved_at"]
        self.return_fields = list(return_fields)
        self.feature_view_name = cfg.feature_view_name
        self.features = [f"{cfg.feature_view_name}:{cfg.embedding_col}"] + [
            f"{cfg.feature_view_name}:{f}" for f in self.return_fields
        ]
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.config = cfg


    def embed_query(self, query_text: str) -> List[float]:
        if not query_text or not query_text.strip():
            raise ValueError("Query text is empty")
        vec = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        return vec[0].astype(np.float32).tolist()

    def get_df_value(self, row, df_columns, field: str):
        pref = f"{self.feature_view_name}__{field}"
        if pref in df_columns:
            return row[pref]
        if field in df_columns:
            return row[field]
        return None

    def search(self, query_text: str, k: int = 5) -> List[SearchHit]:
        query_vec = self.embed_query(query_text)
        res = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_vec,
            top_k=k,
            distance_metric=self.distance_metric
        ).to_df()
        print(list(res.columns))
        score_col = None
        for cand in ("distance", "score", "similarity"):
            if cand in res.columns:
                score_col = cand
                break

        interim: List[Dict[str, Any]] = []
        for _, row in res.iterrows():
            chunk_id = row[self.config.chunk_id_col]
            if chunk_id is None:
                continue
            fields: Dict[str, Any] = {}
            for f in self.return_fields:
                fields[f] = self.get_df_value(row, res.columns, f)

            score = float(row[score_col]) if (score_col and score_col in res.columns) else float("nan")
            interim.append({
                "score": score,
                "chunk_id": str(chunk_id),
                "fields": fields,
            })

        if not interim:
            return []

        s3_to_chunk_ids: Dict[str, Set[str]] = {}
        for it in interim:
            chunk_id = it["chunk_id"]

            base_id = re.sub(r"_\d+$", "", chunk_id)

            s3_uri = f"s3://chunk-docs/web/{base_id}.jsonl"

            s3_to_chunk_ids.setdefault(s3_uri, set()).add(chunk_id)

        chunk_text_map: Dict[str, str] = {}
        for s3_uri, cids in s3_to_chunk_ids.items():
            chunk_text_map.update(fetch_chunk_texts_from_jsonl(self.minio_client, s3_uri, cids))

        out: List[SearchHit] = []
        for it in interim:
            cid = it["chunk_id"]
            out.append(SearchHit(
                score=it["score"],
                chunk_id=cid,
                fields=it["fields"],
                text=chunk_text_map.get(cid, ""),
            ))

        return out