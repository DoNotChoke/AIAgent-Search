from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")
col = Collection("aiagent_docs_docs_embeddings")   # đổi nếu collection tên khác
col.load()

cid = "a2dcb8c913af28e3680c98f1d9d869eb5b6296f75b6a320a712365f0c39faf8f_6"
rows = col.query(
    expr=f'chunk_id == "{cid}"',
    output_fields=["chunk_id", "doc_id", "s3_uri", "url", "retrieved_at"],
    limit=5,
)
print(rows)