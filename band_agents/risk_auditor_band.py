import json
import asyncio
import logging

from band import Agent
from band.agent import SimpleAdapter

from emo import EpisodicMemoryObject
from agents.risk_auditor import run as run_risk_auditor

log = logging.getLogger(__name__)


class RiskAuditorAdapter(SimpleAdapter):
    async def on_message(self, msg, tools, history, participants_msg, contacts_msg, *, is_session_bootstrap, room_id):
        meta = getattr(msg, "metadata", None) or {}
        emo = None
        # Prefer structured EMO in metadata
        if isinstance(meta, dict) and meta.get("emo"):
            emo = EpisodicMemoryObject.from_dict(meta["emo"])
        else:
            try:
                payload = json.loads(msg.content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("emo"):
                emo = EpisodicMemoryObject.from_dict(payload["emo"])
        if emo is None:
            out = {"error": "invalid payload: no emo found in metadata or content"}
            send = getattr(tools, "send_message", None)
            if callable(send):
                await send(room_id, json.dumps(out))
            else:
                log.error("Invalid payload: no emo")
            return

        # Run the risk auditor agent logic
        emo = run_risk_auditor(emo)

        # Store updated EMO
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

        # Supersede previous memory id if present
        prev_mem_id = None
        if isinstance(meta, dict):
            prev_mem_id = meta.get("memory_id")
        if not prev_mem_id:
            try:
                payload = json.loads(msg.content)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                prev_mem_id = payload.get("memory_id")
        if prev_mem_id:
            try:
                await tools.supersede_memory(prev_mem_id)
            except Exception:
                log.debug("Could not supersede previous memory %s", prev_mem_id)

        decision = emo.latest_decision()
        out = {"memory_id": new_mem_id, "decision": decision, "next_agent": "@approval_packager"}
        send = getattr(tools, "send_message", None)
        if callable(send):
            await send(room_id, json.dumps(out))
        else:
            log.info("Response: %s", out)


async def main():
    agent = Agent.from_config("risk_auditor", adapter=RiskAuditorAdapter())
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
