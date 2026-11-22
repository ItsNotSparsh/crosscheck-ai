"""Install only the PyTorch weights and processors used for local photo checks."""
from pathlib import Path
import requests

MODELS = {
    "vilt-vqa": ("dandelin/vilt-b32-finetuned-vqa","d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08",600_000_000),
    "blip-itm": ("Salesforce/blip-itm-base-coco","bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",1_100_000_000),
}
FILES = ["config.json","preprocessor_config.json","tokenizer.json","tokenizer_config.json",
         "special_tokens_map.json","vocab.txt","pytorch_model.bin"]


def main():
    for folder,(repo,rev,cap) in MODELS.items():
        path=Path(__file__).resolve().parent/"artifacts"/folder
        path.mkdir(exist_ok=True)
        for name in FILES:
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
