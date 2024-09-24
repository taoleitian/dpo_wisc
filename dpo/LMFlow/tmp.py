import json
import sys

file_name = sys.argv[1]

data = json.load(open(file_name, 'r'))
res = {}
for d in data:
    res[list(d.keys())[0]] = list(d.values())[0]

json.dump(res, open(file_name, 'w'))
