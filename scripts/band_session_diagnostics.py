import os
import yaml
import requests

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'agent_config.yaml')


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_key(api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://app.band.ai/api/v1/agent/me"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        return {"status_code": r.status_code, "ok": r.ok, "text": r.text}
    except Exception as e:
        return {"status_code": None, "ok": False, "error": str(e)}


def main():
    cfg = load_config(CONFIG_PATH) or {}
    print('Checking agent API keys against Band `/api/v1/agent/me`')
    for agent, info in cfg.items():
        key = info.get('api_key') or ''
        if not key:
            print(f"{agent}: no api_key configured")
            continue
        res = check_key(key)
        if res.get('ok'):
            print(f"{agent}: OK (HTTP {res['status_code']})")
        else:
            if res.get('status_code') is None:
                print(f"{agent}: ERROR ({res.get('error')})")
            else:
                print(f"{agent}: HTTP {res['status_code']} - likely invalid or evicted")
    print('\nIf keys are valid but sessions are being evicted, stop other clients that use the same keys or rotate the keys using `scripts/update_agent_keys.py`.')

if __name__ == '__main__':
    main()
