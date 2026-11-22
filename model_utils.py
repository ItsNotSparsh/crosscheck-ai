from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

def resolve_model_source():
    """Prefer a complete local HF snapshot; otherwise let Transformers download it."""
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots"
    if hub.exists():
        candidates=[]
        for snap in hub.iterdir():
            if snap.is_dir() and (snap / "config.json").is_file() and (snap / "preprocessor_config.json").is_file() and ((snap / "pytorch_model.bin").is_file() or (snap / "model.safetensors").is_file()):
                candidates.append(snap)
        if candidates:
            return str(max(candidates, key=lambda p:(p / "pytorch_model.bin").stat().st_size if (p / "pytorch_model.bin").exists() else (p / "model.safetensors").stat().st_size))
    return MODEL_ID

def load_clip(device):
    source=resolve_model_source()
    local=Path(source).exists()
    proc=CLIPProcessor.from_pretrained(source,local_files_only=local,use_fast=False)
    model=CLIPModel.from_pretrained(source,local_files_only=local).to(device).eval()
    return proc,model
