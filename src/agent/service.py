import argparse
import asyncio
import uvicorn
import contextlib
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from feast import FeatureStore
from langchain.agents import create_agent

from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer

from agent.generator import AgentGeneratorService
from storage.caching import SemanticCache
from storage.config import config
from storage.utils import get_minio_client
from tools.mcp_tools import get_docs_tools, get_web_tools

from typing import Optional

from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_tools = await get_docs_tools(stateful=False)
    web_tools = await get_web_tools(stateful=False)
    tools = docs_tools + web_tools

    store = FeatureStore(repo_path=config.feast_cache_repo)
    minio = get_minio_client(
        config.minio_endpoint,
        config.minio_access_key,
        config.minio_secret_key,
        config.minio_secure
    )
    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    cache = SemanticCache(store=store, encoder=encoder, threshold=0.8, minio_client=minio)

    agent = create_agent(
        model="gpt-5-mini",
        tools=tools,
    )

    app.state.svc = AgentGeneratorService(agent=agent, cache=cache)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/streaming")
async def rag_stream(
    request: Request,
    q: str = Query(...),
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    svc: AgentGeneratorService = request.app.state.svc

    async def event_gen():
        try:
            async for evt in svc.generate_sse(q, session_id=session_id, user_id=user_id):
                if await request.is_disconnected():
                    break
                yield evt
        except asyncio.CancelledError:
            return
        except GeneratorExit:
            return
        except Exception as e:
            with contextlib.suppress(Exception):
                yield f"event: error\ndata: {str(e)}\n\n"
            return

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")