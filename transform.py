import json

with open("train_set/annotations.json") as f:
    data = json.load(f)

new = dict()
for d in data.items():
    new["train_set/download_set/" + d[0]] = d[1]

with open("train_set/annotations.json", "w") as f:
    json.dump(new, f)
