import contextlib
import json
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator

from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

from storage.caching import SemanticCache


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


async def _stream_words_as_sse(text: str, delay_sec: float = 0.0):
    text = (text or "").strip()
    if not text:
        return

    words = text.split()
    for i, w in enumerate(words):
        chunk = w + (" " if i < len(words) - 1 else "")
        yield chunk
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)


class AgentGeneratorService:
    def __init__(
        self,
        agent: Runnable,
        prompt_name: str = "aiagent-search",
        prompt_label: str = "production",
        cache: Optional[SemanticCache] = None,
    ):
        self.agent = agent
        self.langfuse = get_client()
        self.prompt = self.langfuse.get_prompt(prompt_name, label=prompt_label, type="chat")
        self.cache = cache
        self.cache_write_lock = asyncio.Lock()

    def update_trace_context(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session_id or user_id:
            md = {"prompt_version": getattr(self.prompt, "version", None)}
            if extra_metadata:
                md.update(extra_metadata)

            self.langfuse.update_current_trace(
                session_id=session_id,
                user_id=user_id,
                metadata=md,
            )

    def make_inputs(self, question: str) -> Dict[str, Any]:
        messages: List[BaseMessage] = self.prompt.get_langchain_prompt(question=question)
        return {"messages": messages}

    def cache_lookup(self, question: str) -> Optional[str]:
        if not self.cache:
            return None
        try:
            # Expect SemanticCache.search_cache() returns plain text
            return self.cache.search_cache(question)
        except Exception:
            return None

    async def cache_upload(self, question: str, answer: str) -> None:
        if not self.cache:
            return
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return

        async with self.cache_write_lock:
            try:
                await asyncio.to_thread(self.cache.cache_message, question, answer)
            except Exception:
                pass

    @observe(name="aiagent-search_generate")
    async def generate(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        self.update_trace_context(session_id=session_id, user_id=user_id)

        cached = self.cache_lookup(question)
        if cached:
            return cached

        handler = CallbackHandler()
        out = await self.agent.ainvoke(
            self.make_inputs(question),
            config={"callbacks": [handler]},
        )

        if isinstance(out, dict) and "messages" in out and out["messages"]:
            last = out["messages"][-1]
            answer = _as_text(getattr(last, "content", ""))
        else:
            answer = _as_text(getattr(out, "content", out))

        asyncio.create_task(self.cache_upload(question, answer))
        return answer

    @observe(name="aiagent-search_generate_stream")
    async def generate_sse(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        heartbeat_sec: float = 15.0,
    ) -> AsyncGenerator[str, None]:
        self.update_trace_context(session_id=session_id, user_id=user_id)

        yield sse_event(
            "trace",
            {
                "prompt_version": getattr(self.prompt, "version", None),
                "session_id": session_id,
                "user_id": user_id,
            },
        )

        # cache hit => stream from cached final text
        cached = self.cache_lookup(question)
        if cached:
            cached_text = _as_text(cached)
            # stream words
            async for w in _stream_words_as_sse(cached_text, delay_sec=0.02):
                yield sse_event("token", {"text": w})
            yield sse_event("final", {"text": cached_text})
            return

        handler = CallbackHandler()
        inputs = self.make_inputs(question)

        stop_ping = asyncio.Event()
        ping_queue: List[str] = []

        async def ping_loop():
            while not stop_ping.is_set():
                await asyncio.sleep(heartbeat_sec)
                if not stop_ping.is_set():
                    ping_queue.append(sse_event("ping", {}))

        ping_task = asyncio.create_task(ping_loop())

        # We'll stream tokens for UI, but cache ONLY the final assistant message captured at end.
        streamed_parts: List[str] = []
        final_answer: str = ""

        try:
            if hasattr(self.agent, "astream_events"):
                async for ev in self.agent.astream_events(
                    inputs,
                    version="v2",
                    config={"callbacks": [handler], "recursion_limit": 50},
                ):
                    while ping_queue:
                        yield ping_queue.pop(0)

                    etype = ev.get("event")
                    data = ev.get("data") or {}

                    # stream tokens
                    if etype in ("on_chat_model_stream", "on_llm_stream"):
                        chunk = data.get("chunk")
                        if chunk is not None:
                            t = _as_text(getattr(chunk, "content", ""))
                            if t:
                                streamed_parts.append(t)
                                yield sse_event("token", {"text": t})
                        continue

                    # capture final output message for caching
                    if etype in ("on_chain_end", "on_agent_end"):
                        out = data.get("output") or data.get("outputs") or {}
                        if isinstance(out, dict) and "messages" in out and out["messages"]:
                            last = out["messages"][-1]
                            t = _as_text(getattr(last, "content", ""))
                            if t:
                                final_answer = t
                        continue
            else:
                # fallback: astream (may be cumulative)
                prev_full = ""
                async for chunk in self.agent.astream(
                    inputs,
                    config={"callbacks": [handler], "recursion_limit": 50},
                ):
                    while ping_queue:
                        yield ping_queue.pop(0)

                    if isinstance(chunk, dict) and "messages" in chunk and chunk["messages"]:
                        last = chunk["messages"][-1]
                        full = _as_text(getattr(last, "content", ""))
                    elif hasattr(chunk, "content"):
                        full = _as_text(getattr(chunk, "content", ""))
                    else:
                        full = _as_text(chunk)

                    if not full:
                        continue

                    # delta streaming if cumulative
                    if full.startswith(prev_full):
                        delta = full[len(prev_full):]
                    else:
                        delta = full
                    prev_full = full

                    if delta:
                        streamed_parts.append(delta)
                        yield sse_event("token", {"text": delta})

                final_answer = prev_full

            # IMPORTANT: cache only final_answer (last assistant message), not streamed_parts
            answer = (final_answer.strip() if final_answer else "".join(streamed_parts)).strip()

            yield sse_event("final", {"text": answer})
            asyncio.create_task(self.cache_upload(question, answer))

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

        finally:
            stop_ping.set()
            ping_task.cancel()
            with contextlib.suppress(Exception):
                await ping_task