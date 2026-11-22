"""Exercise the normal Flask route on fixed external examples. No training."""
import io
import json
import time
from collections import Counter
from pathlib import Path

from app import app
from live_screening import UNCERTAIN

DATA = Path(__file__).resolve().parent / "data" / "showcase_eval"
EXPECTED = {"SUPPORTS":"Real","REFUTES":"Fake","NOT ENOUGH INFO":UNCERTAIN}
PHOTO = [
    ("coco_cats.png","A cat is lying on a sofa","Real"),
    ("coco_cats.png","A man is sitting on a sofa","Fake"),
    ("astronaut.png","A person","Real"), ("astronaut.png","A giraffe","Fake"),
    ("coffee.png","A cup","Real"), ("coffee.png","A bicycle","Fake"),
]


def main():
    rows = []
    client = app.test_client()
    def run(group, mode, claim, gold, photo=None):
        data = {"text":claim,"mode":mode}
        if photo:
            data["image"] = (io.BytesIO((DATA/photo).read_bytes()), photo)
        start = time.monotonic()
        response = client.post("/api/analyze",data=data)
        result = response.get_json()
        pred = result.get("label","ERROR")
        rows.append(dict(group=group, mode=mode, claim=claim, expected=gold, predicted=pred,
                         basis=result.get("decision_basis",result.get("error")), photo=photo,
                         sources=(result.get("evidence") or {}).get("items",[]),
                         seconds=round(time.monotonic()-start,2)))
        print(group, len(rows), gold, pred, flush=True)
    for file, claim, label in PHOTO:
        run("photo-baseline","dataset",claim,label,file)
        run("photo-new","auto",claim,label,file)
    for row in json.loads((DATA/"fever_holdout.json").read_text()):
        run("fever","news",row["claim"],EXPECTED[row["label"]])
    for claim,label in [("Narendra Modi is the Prime Minister of India","Real"),
                        ("Narendra Modi is not the Prime Minister of India","Fake")]:
        run("office","auto",claim,label)
    for group in dict.fromkeys(row["group"] for row in rows):
        subset=[row for row in rows if row["group"]==group]
        print(group, len(subset),sum(r["predicted"]==r["expected"] for r in subset),
              Counter(r["predicted"] for r in subset))
    (DATA/"results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
