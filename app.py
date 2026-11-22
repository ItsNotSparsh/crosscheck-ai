"""Flask interface for CrossCheck AI's CLIP + Fakeddit classifier."""
from pathlib import Path
import io
import warnings

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from model_utils import load_clip as load_clip_model
from evidence_search import retrieve_evidence
from evidence_verification import assess_claim
from live_screening import check_news, check_photo, photo_questions, INFERENCE_LOCK

ROOT=Path(__file__).resolve().parent
ARTIFACT=ROOT/"artifacts"/"clip_fakeddit_classifier.joblib"
LABELS={0:"Fake",1:"Real",2:"Possibly misleading / uncertain"}
# Conservative disagreement gate: a weak Fake prediction should not overrule
# a reasonably aligned image/claim pair. Similarity alone never proves truth.
FAKE_ALIGNMENT_ABSTAIN_CONFIDENCE=.65
FAKE_ALIGNMENT_SIMILARITY=.28
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=13*1024*1024
warnings.simplefilter("error", Image.DecompressionBombWarning)
_clip=None
_bundle=None

def get_models():
    global _clip,_bundle
    if _clip is None:
        processor,model=load_clip_model(DEVICE)
        _clip=(processor,model)
    if _bundle is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError("Trained classifier not found. Run the training command in README.md first.")
        _bundle=joblib.load(ARTIFACT)
    return _clip[0],_clip[1],_bundle

def make_features(text,image,processor,model):
    image=image.convert("RGB")
    batch=processor(text=[text[:1000] or " "," "],images=[image,image],return_tensors="pt",padding=True,truncation=True)
    batch={key:value.to(DEVICE) for key,value in batch.items()}
    with torch.inference_mode():
        output=model(**batch)
        t=torch.nn.functional.normalize(output.text_embeds,dim=-1)
        v=torch.nn.functional.normalize(output.image_embeds,dim=-1)
        similarity=float((t[0]*v[0]).sum().item())
        vector=torch.cat([t[0],v[0],t[0]*v[0],torch.tensor([similarity],device=DEVICE)]).cpu().numpy()
    return vector.reshape(1,-1),similarity

@app.get("/")
def index():
    return render_template("index.html",gpu=("NVIDIA CUDA" if DEVICE=="cuda" else "CPU"))

@app.get("/api/health")
def health():
    return jsonify({"ok":True,"device":DEVICE,"model_ready":ARTIFACT.exists()})

@app.post("/api/analyze")
def analyze():
    mode=request.form.get("mode","auto")
    if mode not in {"auto","news","photo","dataset"}:
        return jsonify({"error":"Unknown analysis mode."}),400
    claim=request.form.get("text","").strip()
    uploaded=request.files.get("image")
    if len(claim)<3:
        return jsonify({"error":"Enter the post or claim text (at least 3 characters)."}),400
    if len(claim)>1000:
        return jsonify({"error":"Keep the claim under 1,001 characters."}),400
    has_image=bool(uploaded and uploaded.filename)
    if mode=="auto":
        mode="photo" if has_image and photo_questions(claim) else "news"
    if not has_image and mode in {"photo","dataset"}:
        return jsonify({"error":"This check requires an image."}),400
    image=None
    try:
        if has_image:
            image_bytes=uploaded.read()
            if len(image_bytes)>12*1024*1024:
                return jsonify({"error":"Choose an image under 12 MB."}),400
            image=Image.open(io.BytesIO(image_bytes))
            if image.width*image.height > 25_000_000:
                return jsonify({"error":"Image dimensions are unusually large. Resize it and try again."}),400
            image.verify()
            image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        return jsonify({"error":"That file could not be read as an image. Try PNG, JPG, or WEBP."}),400
    try:
        if mode=="photo":
            result=check_photo(claim,image)
            return jsonify({**result,"mode":"photo","evidence":None})
        if mode=="news":
            result=check_news(claim)
            result["mode"]="news"
            if has_image:
                with INFERENCE_LOCK:
                    processor,model,_=get_models()
                    _,result["similarity"]=make_features(claim,image,processor,model)
                result["reasons"].append("The verdict covers the written claim. CLIP compares the attached image with the wording. Image date and origin remain unverified.")
            if request.form.get("evidence")=="true":
                leads=retrieve_evidence(claim)
                for item in leads["items"]:
                    if item["url"] not in {r["url"] for r in result["evidence"]["items"]}:
                        result["evidence"]["items"].append({**item,"title":"Related link · "+item["title"]})
            return jsonify(result)
        with INFERENCE_LOCK:
            processor,model,bundle=get_models()
            features,similarity=make_features(claim,image,processor,model)
        classifier=bundle["multimodal"]
        probabilities=classifier.predict_proba(features)[0]
        classes=np.asarray(classifier.classes_,dtype=int)
        sorted_indices=np.argsort(probabilities)[::-1]
        raw_class=int(classes[sorted_indices[0]])
        confidence=float(probabilities[sorted_indices[0]])
        threshold=float(bundle.get("confidence_threshold",.55))
        low_confidence=confidence<threshold
        alignment_conflict=(raw_class==0 and confidence<FAKE_ALIGNMENT_ABSTAIN_CONFIDENCE
                            and similarity>=FAKE_ALIGNMENT_SIMILARITY)
        abstained=low_confidence or alignment_conflict
        predicted=2 if abstained else raw_class
        reasons=[]
        if low_confidence:
            reasons.append(f"Confidence was below the validation-selected threshold ({threshold:.0%}), so the app abstained to the uncertain category.")
        if alignment_conflict:
            reasons.append("The Fake signal was not decisive and the image had reasonable semantic alignment with the claim, so the model abstained instead of treating its dataset pattern as a verdict.")
        else:
            if not low_confidence:
                reasons.append(f"The Fakeddit-trained classifier's highest calibrated probability was {LABELS[raw_class]} ({confidence:.0%}).")
        reasons.append(f"CLIP text-image cosine similarity is {similarity:.3f}. This indicates semantic alignment only; it does not verify truth or image date/source.")
        show_sources=request.form.get("evidence")=="true"
        result={"label":LABELS[predicted],"raw_label":LABELS[raw_class],"abstained":abstained,"confidence":confidence,"classifier_confidence":confidence,"decision_score":confidence,"decision_basis":"model abstained due to image/classifier disagreement" if alignment_conflict else "Fakeddit-trained multimodal classifier","threshold":threshold,"similarity":similarity,"probabilities":[{"label":LABELS[int(c)],"value":float(p)} for c,p in sorted(zip(classes,probabilities),key=lambda x:-x[1]) if int(c) in LABELS],"reasons":reasons,"device":DEVICE}
        # Keep optional source leads for review; they do not replace the
        # first-build multimodal classifier's prediction.
        evidence=retrieve_evidence(claim) if show_sources else {"items":[],"message":"Optional links are off."}
        items=evidence.get("items",[])
        check={"stance":"unavailable","supports":0.0,"refutes":0.0,"checked":0}
        if items:
            try:
                check=assess_claim(claim,items)
            except Exception as exc:
                app.logger.exception("Evidence comparison failed")
                check={"stance":"unavailable","supports":0.0,"refutes":0.0,"checked":0,"message":type(exc).__name__}
        result["evidence_check"]=check
        result["mode"]="dataset"
        result["evidence"]={**evidence,"items":items if show_sources else []}
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify({"error":f"Analysis could not be completed: {type(exc).__name__}. Check the project setup and try again."}),500

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error":"The upload request exceeded the 12 MB limit. If your file is smaller, restart the app and try again."}),413

if __name__=="__main__":
    app.run(host="127.0.0.1",port=8501,debug=False,threaded=True)
