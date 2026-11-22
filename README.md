# CrossCheck AI — Multimodal Fake News Screening

An Applications of AI course project that screens a social-media claim together with its associated image. It uses pretrained CLIP ViT-B/32 embeddings and a calibrated supervised classifier trained on Fakeddit. The output is **Fake**, **Real**, or **Possibly misleading / uncertain**. It is an educational signal for human review, not a fact-check verdict.

## Project features

- Custom responsive HTML/CSS/JavaScript interface served by Flask (no Streamlit).
- CLIP text and image encoders with a lightweight supervised fusion classifier.
- Probability calibration and a confidence threshold selected from a held-out slice of training data; the official test split is kept separate for final evaluation.
- External search and a pretrained NLI model compare retrieved text with the claim on every analysis. The checkbox controls whether clickable source links are shown; it does not disable the source check.
- Text-image semantic similarity, class scores, explanatory notes, and source links to inspect.
- Image size/type validation and inference on CUDA when available.

## Run the app (Windows)

From `C:\Users\spars\Desktop\AAI_Project`:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Or double-click `run_demo.bat`. Open <http://127.0.0.1:8501>. The app needs the trained classifier at `artifacts/clip_fakeddit_classifier.joblib`. It loads CLIP and the DeBERTa NLI model from the Hugging Face cache; first use downloads these pretrained models. The detected hardware in this environment is an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM.

## Setup from scratch

```powershell
cd "C:\Users\spars\Desktop\AAI_Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build matching your driver if `torch.cuda.is_available()` returns false. Model card: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

## Dataset and training

The project uses Fakeddit, the paired social-media text/image dataset identified in the milestone report. A 73.9 MB parquet mirror is in `data/fakeddit_train.parquet`; the full image archive is over 100 GB, so `prepare_data.py` downloads only selected image URLs and preserves the dataset's `general_train` and `general_test` partitions.

Current subset: **5,686 paired training examples** and **1,432 held-out test examples**. Images are in `data/images/`; the split manifests are `data/sample_train.tsv` and `data/sample_test.tsv`.

```powershell
python prepare_data.py data\fakeddit_train.parquet data\sample_train.tsv data\images --split train --limit 6000
python prepare_data.py data\fakeddit_train.parquet data\sample_test.tsv data\images --split test --limit 1500
python train.py --train data\sample_train.tsv --test data\sample_test.tsv --images data\images --labels 6-way --limit 6000 --batch-size 16
```

Training keeps CLIP frozen and fits the classifier over embeddings. It does not fine-tune the full transformer, which keeps GPU memory use practical on a 6 GB RTX 4050. If memory is tight, reduce `--batch-size` to 8 or 4. Training writes `artifacts/clip_fakeddit_classifier.joblib` and `artifacts/evaluation.json`.

Fakeddit six-way labels are mapped for this course task as follows: 0 True → Real; 2 False Connection and 4 Manipulated → Fake; 1 Satire, 3 Imposter, and 5 Misleading → Possibly misleading / uncertain. This is a project policy; Fakeddit labels are platform-specific and do not define universal truth.

## Evaluation

The test split is not used to select the confidence threshold. Training examples are split again for probability calibration and threshold selection. Final test metrics are saved in `artifacts/evaluation.json`; run training to regenerate them for your own subset. Report accuracy, macro-F1, per-class results, and the confusion matrix as a dataset-specific measurement, not general-world accuracy.

## Interpretation and limitations

- CLIP cosine similarity estimates semantic alignment. It does not establish truth, image provenance, recency, manipulation, or whether a visual is from the claimed event.
- The classifier learns Fakeddit patterns and inherits its noisy, dated, Reddit-derived labels. Its score is not a probability that a real-world claim is objectively true.
- The final class uses retrieved-text entailment: strong source support maps to **Real**, strong refutation maps to **Fake**, and unavailable, weak, or mixed evidence maps to **Possibly misleading / uncertain**. The Fakeddit classifier remains visible as a separate text-image signal; its percentage is not a probability that the claim is true.
- Search providers can intermittently block or time out. The app tries fallbacks and returns **Possibly misleading / uncertain** if no decisive source comparison is available. Search results and snippets are leads, not verification.
- Open and assess trusted, preferably primary, sources before making a conclusion. Do not use this prototype for high-stakes decisions or to label individuals.

## Course objectives addressed

4. Compare claim and image semantics with normalized CLIP embeddings.
5. Retrieve source text and compare its stance with the claim using a pretrained NLI model.
6. Explain the final evidence-based result alongside classifier scores, similarity, and optional source links.
7. Evaluate on the held-out dataset split using accuracy, macro-F1, per-class metrics, and a confusion matrix.

## References

- Nakamura, Levy & Wang (2020), [r/Fakeddit](https://aclanthology.org/2020.lrec-1.755/).
- Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32).
- [Fakeddit dataset repository](https://github.com/entitize/Fakeddit) and [dataset mirror](https://huggingface.co/datasets/AdoCleanCode/Fakeddit).
