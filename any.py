import json
import os

folder = "data/raw/news"

for f in os.listdir(folder):
    with open(os.path.join(folder, f)) as file:
        data = json.load(file)
        print(f, len(data["raw_text"]))
