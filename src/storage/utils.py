import io
import logging
from minio import Minio

from typing import Tuple, Optional, Dict

from hashlib import sha256

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def stable_doc_id(url: str) -> str:
    return sha256(url.encode("utf-8")).hexdigest()

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    s3_uri = s3_uri[5:]
    if "/" not in s3_uri:
        raise ValueError(f"Invalid S3 URI (missing key): {s3_uri}")
    bucket, key = s3_uri.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri (missing bucket or key): {s3_uri}")
    return bucket, key


def get_minio_client(minio_endpoint: str, minio_access_key: str, minio_secret_key: str, secure: bool = False) -> Minio:
    return Minio(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created bucket %s", bucket)
    else:
        logger.info("Bucket %s already exists", bucket)


def build_s3uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"

def upload_documents(
        client: Minio,
        s3_uri: str,
        payload: str,
        content_type: str = "text/markdown",
        metadata: Optional[Dict[str, str]] = None,
):
    bucket, key = parse_s3_uri(s3_uri)
    b = payload.encode("utf-8")
    bio = io.BytesIO(b)
    client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=bio,
        length=len(b),
        content_type=content_type,
        metadata=metadata,
    )

def download_object(
        client: Minio,
        s3_uri: str,
) -> Tuple[bytes, Optional[Dict[str, str]]]:
    bucket, key = parse_s3_uri(s3_uri)

    try:
        st = client.stat_object(bucket, key)
        meta = st.metadata or {}
    except Exception:
        meta = None
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return data, meta


def download_text(
    client: Minio,
    s3_uri: str,
    encoding: str = "utf-8",
) -> Tuple[str, Optional[Dict[str, str]]]:
    b, meta = download_object(client, s3_uri)
    return b.decode(encoding, errors="replace"), meta
