"""Install only the PyTorch weights and processors used for local photo checks."""
from pathlib import Path
import requests

MODELS = {
    "passage-ranker": ("cross-encoder/ms-marco-MiniLM-L6-v2","233902d25c440f23af6f7d6e94d2946bac0bee0a",100_000_000),
    "vilt-vqa": ("dandelin/vilt-b32-finetuned-vqa","d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08",600_000_000),
    "blip-itm": ("Salesforce/blip-itm-base-coco","bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",1_100_000_000),
}
FILES = ["config.json","preprocessor_config.json","tokenizer.json","tokenizer_config.json",
         "special_tokens_map.json","vocab.txt","pytorch_model.bin"]


def main():
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    detector = Path(__file__).resolve().parent / "artifacts" / "fasterrcnn_v2.pth"
    detector.parent.mkdir(exist_ok=True)
    if not detector.exists():
        torch.hub.download_url_to_file(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.url,
                                       str(detector), hash_prefix="dd69338a")
    for folder,(repo,rev,cap) in MODELS.items():
        path=Path(__file__).resolve().parent/"artifacts"/folder
        path.mkdir(exist_ok=True)
        files = [n for n in FILES if n not in {"preprocessor_config.json","pytorch_model.bin"}]+["model.safetensors"] if folder=="passage-ranker" else FILES
        for name in files:
            dst=path/name
            if dst.is_file():
                cap-=dst.stat().st_size
                continue
            partial=dst.with_suffix(".part")
            try:
                url=f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
                with requests.get(url,stream=True,timeout=(10,60)) as r, partial.open("wb") as out:
                    r.raise_for_status()
                    for chunk in r.iter_content(1024*1024):
                        cap-=len(chunk)
                        if cap<0:
                            raise ValueError("Model size limit exceeded")
                        out.write(chunk)
                partial.replace(dst)
                print(folder,name,dst.stat().st_size,flush=True)
            finally:
                partial.unlink(missing_ok=True)


if __name__=="__main__":
    main()
