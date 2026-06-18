import yaml
import os
from getpass import getpass

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'agent_config.yaml')

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    cfg = load_config(CONFIG_PATH) or {}
    print('Current agents:')
    for k in cfg.keys():
        print(' -', k)

    agent = input('Agent to update (or ENTER to quit): ').strip()
    if not agent:
        print('No changes made.')
        return
    if agent not in cfg:
        print('Unknown agent:', agent)
        return

    print('Enter the new API key for', agent)
    print('It will be stored in', CONFIG_PATH)
    new_key = getpass('New API key (input hidden): ')
    if not new_key:
        print('No key entered, aborting')
        return

    cfg[agent]['api_key'] = new_key
    save_config(CONFIG_PATH, cfg)
    print('Updated api_key for', agent)

if __name__ == '__main__':
    main()
