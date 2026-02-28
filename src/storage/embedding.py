from typing import List, Sequence
from langchain_text_splitters import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer

from storage.utils import stable_doc_id

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    url: str
    idx: int
    text: str

def iter_batches(items: Sequence[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)


def splitting(url: str, payload: str) -> List[Chunk]:
    doc_id = stable_doc_id(url)
    splits = splitter.split_text(payload)
    chunks: List[Chunk] = []
    for i, chunk in enumerate(splits, start=1):
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{i}",
            url=url,
            idx=i,
            text=chunk,
        ))
    return chunks


def embedding(model: SentenceTransformer, chunks: List["Chunk"], batch_size: int = 64) -> List[List[float]]:
    """
    Encode chunks to embeddings in batches.

    Returns:
        List[List[float]] with same order as `chunks`.
    """
    texts = [c.text for c in chunks if c and getattr(c, "text", None)]
    if not texts:
        return []

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vecs.tolist()

