"""Train/evaluate the final frozen-CLIP multimodal classifier from Fakeddit TSVs.

Expected Fakeddit TSV columns: id, clean_title (or title), image_id (optional), and
one of 2_way_label, 3_way_label, 6_way_label.
Fakeddit six-way IDs: 0 true, 1 satire, 2 false connection, 3 imposter,
4 manipulated, 5 misleading. Mapping is documented below.
If using Fakeddit 3-way: 0 true, 1 satire, 2 false.
"""
import argparse, io
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from model_utils import load_clip

def map_label(v, mode):
    n=int(v)
    if mode == "2-way": return 1 if n == 0 else 0
    if mode == "3-way": return {0:1,1:2,2:0}[n]
    # Real only means True. Satire, imposter and misleading map to the middle
    # class; false-connection and manipulated-content map to Fake.
    return {0:1,1:2,2:0,3:2,4:0,5:2}[n]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="Fakeddit train TSV")
    ap.add_argument("--test", required=True, help="Fakeddit validation/test TSV")
    ap.add_argument("--images", required=True, help="Folder containing images named by id or image_id")
    ap.add_argument("--labels", choices=["2-way","3-way","6-way"], default="6-way")
    ap.add_argument("--limit", type=int, default=4000, help="Rows per split; start small on a laptop")
    ap.add_argument("--batch-size", type=int, default=16)
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    proc,model=load_clip(device)
    def load(path):
        df=pd.read_csv(path,sep="\t").head(args.limit)
        text_col="clean_title" if "clean_title" in df else ("title" if "title" in df else "text")
        label_col={"2-way":"2_way_label","3-way":"3_way_label","6-way":"6_way_label"}[args.labels]
        if label_col not in df and "label" in df: label_col="label"
        if label_col not in df: raise ValueError(f"Missing {label_col}; columns: {list(df.columns)}")
        records=[]
        for _,r in df.iterrows():
            if pd.isna(r.get(text_col)) or pd.isna(r.get(label_col)): continue
            stem=str(r.get("image_id",r.get("id","")))
            candidates=[Path(args.images)/f"{stem}{ext}" for ext in (".jpg",".jpeg",".png",".webp")]
            if not stem or stem=="nan": stem=str(r.get("id","")); candidates=[Path(args.images)/f"{stem}{ext}" for ext in (".jpg",".jpeg",".png",".webp")]
            img_path=next((p for p in candidates if p.exists()),None)
            if not img_path: continue
            try:
                with Image.open(img_path) as im: im.verify()
            except Exception: continue
            records.append((str(r[text_col])[:1000],img_path,map_label(r[label_col],args.labels)))
        feats=[]; ys=[]
        print(f"  Found {len(records)} usable paired records in {Path(path).name}",flush=True)
        for start in range(0,len(records),args.batch_size):
            chunk=records[start:start+args.batch_size]
            images=[]; valid=[]
            for record in chunk:
                try:
                    with Image.open(record[1]) as im: images.append(im.convert("RGB"))
                    valid.append(record)
                except Exception: continue
            if not valid: continue
            inp=proc(text=[z[0] for z in valid],images=images,return_tensors="pt",padding=True,truncation=True)
            inp={k:v.to(device) for k,v in inp.items()}
            with torch.inference_mode(): out=model(**inp)
            t=torch.nn.functional.normalize(out.text_embeds,dim=-1).cpu().numpy()
            im=torch.nn.functional.normalize(out.image_embeds,dim=-1).cpu().numpy()
            s=np.sum(t*im,axis=1,keepdims=True)
            feats.extend(np.concatenate([t,im,t*im,s],axis=1))
            ys.extend(z[2] for z in valid)
            if (start//args.batch_size+1)%25==0 or start+args.batch_size>=len(records):
                print(f"  Encoded {min(start+args.batch_size,len(records))}/{len(records)}",flush=True)
            del inp,out,images
        return np.asarray(feats),np.asarray(ys)
    print("Extracting train features..."); Xtr,ytr=load(args.train)
    print("Extracting evaluation features..."); Xte,yte=load(args.test)
    if len(ytr)==0 or len(yte)==0: raise SystemExit("No usable paired images found; check --images folder and TSV ids.")
    # Keep a stratified slice of training data for threshold selection. Calibrate
    # probabilities on CV folds of the remaining training data only; never use test labels.
    Xfit,Xval,yfit,yval=train_test_split(Xtr,ytr,test_size=.2,random_state=42,stratify=ytr)
    base=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,class_weight="balanced",C=.5))
    clf=CalibratedClassifierCV(base,method="sigmoid",cv=3)
    clf.fit(Xfit,yfit)
    pv=clf.predict_proba(Xval); classes=np.asarray(clf.classes_)
    confidence=pv.max(axis=1); base_pred=classes[pv.argmax(axis=1)]
    thresholds=np.arange(.35,.851,.025)
    scores=[]
    for threshold in thresholds:
        pred_val=base_pred.copy(); pred_val[confidence<threshold]=2
        scores.append(f1_score(yval,pred_val,average="macro"))
    conf_threshold=float(thresholds[int(np.argmax(scores))])
    pred=clf.predict(Xte); pte=clf.predict_proba(Xte); conf_test=pte.max(axis=1)
    pred[conf_test<conf_threshold]=2
    metrics={"accuracy":float(accuracy_score(yte,pred)),"macro_f1":float(f1_score(yte,pred,average="macro")),"report":classification_report(yte,pred,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(yte,pred).tolist()}
    print(f"Multimodal CLIP: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    out=Path("artifacts"); out.mkdir(exist_ok=True)
    metrics["confidence_threshold"]=conf_threshold
    metrics["validation_macro_f1_after_abstention"]=float(max(scores))
    joblib.dump({"multimodal":clf,"metrics":metrics,"confidence_threshold":conf_threshold,"similarity_threshold":None,"label_mapping":args.labels,"calibrated":True},out/"clip_fakeddit_classifier.joblib")
    import json
    (out/"evaluation.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"Saved model and metrics in {out.resolve()}")
if __name__=="__main__": main()
