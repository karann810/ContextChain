import json, os
errs=False
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.json'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    json.load(fh)
            except Exception as e:
                print(p + ': ' + str(e))
                errs=True
if not errs:
    print('No JSON errors found')
