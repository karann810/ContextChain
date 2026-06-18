import json
import uuid
import asyncio
import logging

from band import Agent
from band.agent import SimpleAdapter

from core.emo import EpisodicMemoryObject
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

        decision = emo.latest_decision()
        out = {"emo": emo.to_dict(), "decision": decision, "next_agent": "@vendor_intelligence"}
        # Try to send a response if tools supports send_message
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
