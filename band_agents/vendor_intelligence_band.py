import json
import asyncio
import logging

from band import Agent
from band.agent import SimpleAdapter

from emo import EpisodicMemoryObject
from agents.vendor_intelligence import run as run_vendor_intel

log = logging.getLogger(__name__)


class VendorIntelligenceAdapter(SimpleAdapter):
    async def on_message(self, msg, tools, history, participants_msg, contacts_msg, *, is_session_bootstrap, room_id):
        meta = getattr(msg, "metadata", None) or {}
        emo = None
        if isinstance(meta, dict) and meta.get("emo"):
            emo = EpisodicMemoryObject.from_dict(meta["emo"])
        else:
            try:
                payload = json.loads(msg.content)
            except Exception as e:
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

        emo = run_vendor_intel(emo)

        confidence = emo.latest_confidence()
        next_agent = "@risk_auditor" if confidence < 0.70 else "@approval_packager"

        # Store updated EMO via Band memory API
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

        # Supersede previous memory id if provided in metadata or payload
        prev_mem_id = None
        meta = getattr(msg, "metadata", None) or {}
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
        out = {"memory_id": new_mem_id, "decision": decision, "next_agent": next_agent}
        send = getattr(tools, "send_message", None)
        if callable(send):
            await send(room_id, json.dumps(out))
        else:
            log.info("Response: %s", out)


async def main():
    agent = Agent.from_config("vendor_intelligence", adapter=VendorIntelligenceAdapter())
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
