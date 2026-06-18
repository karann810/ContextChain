import json
import asyncio
import logging

from band import Agent
from band.agent import SimpleAdapter

from core.emo import EpisodicMemoryObject
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

        decision = emo.latest_decision()
        out = {"emo": emo.to_dict(), "decision": decision, "next_agent": next_agent}
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
