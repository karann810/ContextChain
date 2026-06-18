import asyncio
import importlib
import logging
import sys
import time

MODULES = [
    "band_agents.needs_analyzer_band",
    "band_agents.vendor_intelligence_band",
    "band_agents.risk_auditor_band",
    "band_agents.approval_packager_band",
]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


async def runner(module_name: str):
    backoff = 1
    while True:
        try:
            logging.info("Starting %s", module_name)
            mod = importlib.import_module(module_name)
            main = getattr(mod, "main")
            await main()
            logging.info("%s exited cleanly", module_name)
            return
        except Exception:
            logging.exception("%s crashed; restarting in %s seconds", module_name, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def main():
    tasks = [asyncio.create_task(runner(m)) for m in MODULES]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
        sys.exit(0)
