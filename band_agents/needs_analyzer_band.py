import json
import uuid
import asyncio
import logging

from band import Agent
from band.agent import SimpleAdapter

from emo import EpisodicMemoryObject
from agents.needs_analyzer import run as run_needs_analyzer

log = logging.getLogger(__name__)


class NeedsAnalyzerAdapter(SimpleAdapter):
    async def on_message(self, msg, tools, history, participants_msg, contacts_msg, *, is_session_bootstrap, room_id):
        # msg is a PlatformMessage with .content
        payload = None
        # Prefer structured EMO sent in platform metadata
        meta = getattr(msg, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("emo"):
            emo = EpisodicMemoryObject.from_dict(meta["emo"])
        else:
            try:
                payload = json.loads(msg.content)
            except Exception:
                payload = None

            if isinstance(payload, dict) and payload.get("emo"):
                emo = EpisodicMemoryObject.from_dict(payload["emo"])
            else:
                task_id = str(uuid.uuid4())
                raw_text = meta.get("raw_input") or msg.content
                emo = EpisodicMemoryObject(task_id=task_id, raw_input=raw_text)

        emo = run_needs_analyzer(emo)

        # Store updated EMO as an organization-scoped episodic memory
        try:
            new_mem = await tools.store_memory(
                content=json.dumps(emo.to_dict()),
                system="working",
                type="episodic",
                segment="user",
                thought=f"Agent update: {emo.entries[-1].agent_id} v{emo.emo_version}",
                scope="organization",
                metadata={"task_id": emo.task_id, "emo_version": emo.emo_version},
            )
            new_mem_id = new_mem.get("id") if isinstance(new_mem, dict) else getattr(new_mem, "id", None)
        except Exception as e:
            log.error("Failed to store memory: %s", e)
            new_mem_id = None

        # If incoming payload included a previous memory id, supersede it
        prev_mem_id = None
        meta = getattr(msg, "metadata", None) or {}
        if isinstance(meta, dict):
            prev_mem_id = meta.get("memory_id")
        if not prev_mem_id and isinstance(payload, dict):
            prev_mem_id = payload.get("memory_id")
        if prev_mem_id:
            try:
                await tools.supersede_memory(prev_mem_id)
            except Exception:
                log.debug("Could not supersede previous memory %s", prev_mem_id)

        decision = emo.latest_decision()
        out = {"memory_id": new_mem_id, "decision": decision, "next_agent": "@vendor_intelligence"}
        send = getattr(tools, "send_message", None)
        if callable(send):
            await send(room_id, json.dumps(out))
        else:
            log.info("Response: %s", out)


async def main():
    agent = Agent.from_config("needs_analyzer", adapter=NeedsAnalyzerAdapter())
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
