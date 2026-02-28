from langchain_redis import RedisSemanticCache
from langchain_core.outputs import Generation
from langchain_community.embeddings import SentenceTransformerEmbeddings


class SemanticCache:
    def __init__(
            self,
            redis_url: str,
            embeddings: SentenceTransformerEmbeddings,
            llm_string: str = "aiagent-search_service"
    ):
        self.semantic_cache = RedisSemanticCache(
            redis_url=redis_url,
            embeddings=embeddings,
            distance_threshold=0.1,
            ttl=60
        )
        self.llm_string = llm_string

    def cache_message(self, question: str, response: str):
        self.semantic_cache.update(
            prompt=question,
            llm_string=self.llm_string,
            return_val=[
                Generation(
                    text=str(response)
                )
            ]
        )

    def search_cache(self, question: str):
        cached_response = self.semantic_cache.lookup(question, self.llm_string)
        print(f"Cached response: {cached_response}")
        if cached_response:
            return cached_response[0].text
        return None
