"""Exercise the normal Flask route on fixed external examples. No training."""
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    if "--release" in sys.argv:
        def check(row):
            response=app.test_client().post("/api/analyze",data={"text":row["claim"],"mode":"auto"})
            result=response.get_json()
            if response.status_code!=200:
                raise RuntimeError(result)
            return dict(**row,predicted=result["label"],basis=result["decision_basis"],sources=result["evidence"]["items"])
        manifest=json.loads((DATA/"release_holdout.json").read_text(encoding="utf8"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(check,manifest))
        (DATA/"release_results.json").write_text(json.dumps(results,indent=2),encoding="utf8")
        print("Completed release comparison",len(results))
        return
    if any(flag in sys.argv for flag in ("--averitec","--external","--paired","--regression")):
        labels = {"Supported":"Real","Refuted":"Fake","Not Enough Evidence":UNCERTAIN,
                  "Conflicting Evidence/Cherrypicking":UNCERTAIN}
        name = next((k for k in ("paired","external","regression") if "--"+k in sys.argv),"averitec")
        manifest = "external" if name=="regression" else name
        for row in json.loads((DATA/f"{manifest}_holdout.json").read_text(encoding="utf8")):
            run("averitec","news",row["claim"],labels[row["expected"]])
        (DATA/f"{name}_results.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
        return
    if "--photo" in sys.argv:
        for file,claim,label in PHOTO:
            run("photo-new","auto",claim,label,file)
        for pet in json.loads((DATA/"pets"/"manifest.json").read_text()):
            run("pets","auto","A "+pet["species"],"Real","pets/"+pet["file"])
        (DATA/"photo_results.json").write_text(json.dumps(rows,indent=2),encoding="utf8")
        return
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
