# CrossCheck AI — checking claims and photo captions

Enter a claim in the Flask app and choose Automatic. Factual claims can be text-only. For a visible caption, attach a JPG, PNG or WEBP. Select Visible caption explicitly for longer descriptions. Select Original Fakeddit classifier to reproduce the earlier dataset-based workflow.

## How the current project works

- **Public claims.** CrossCheck searches publisher reviews and a local archive of 397,396 original publisher review records plus 3,068 AVeriTeC training records for exact claims and closely matching paraphrases. It preserves numeric qualifications and checks entailment in both directions for paraphrases. The archive contains dated, labeled examples for retrieval. The review history refreshes in the background. Open the linked citation to inspect its scope.
- **Background facts.** The app checks explicitly worded relationships against Wikidata and retrieves passages from live English Wikipedia and from 241,787 locally indexed Simple English Wikipedia articles dated November 2023. A MiniLM passage reranker selects relevant excerpts before DeBERTa checks support or contradiction. Named-entity confusion, dated records and multi-step claims remain sources of error.
- **Current events.** For recognized event wording, the app searches for dated publisher articles and requires corroboration from two domains. A dedicated lookup uses recent statements from the Indian Prime Minister's Office for explicit office-holder claims. Unavailable sources, missing media identity or inconclusive evidence can yield Possibly misleading / uncertain. Extra related links are optional.
- **Photos.** Faster R-CNN ResNet-50 FPN V2 checks presence claims about common objects. ViLT answers visual questions and BLIP assesses caption fit. A claim about a person holding office or about where a photo was taken requires external evidence. The photo mode checks visible content only.

**Linked articles.** Choose **Compare with a news article**, paste a direct supported publisher URL and enter the claim. Read the dated passage alongside the result. Real and Fake in this mode describe agreement with that article. Automatic mode continues to fetch its own sources.

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

## Original classifier training

The project uses Fakeddit, the paired social-media text/image dataset identified in the milestone report. A 73.9 MB parquet mirror is in `data/fakeddit_train.parquet`; the full image archive is over 100 GB, so `prepare_data.py` downloads only selected image URLs and preserves the dataset's `general_train` and `general_test` partitions.

Current subset: **5,686 paired training examples** and **1,432 held-out test examples**. Images are in `data/images/`; the split manifests are `data/sample_train.tsv` and `data/sample_test.tsv`.

```powershell
python prepare_data.py data\fakeddit_train.parquet data\sample_train.tsv data\images --split train --limit 6000
python prepare_data.py data\fakeddit_train.parquet data\sample_test.tsv data\images --split test --limit 1500
python train.py --train data\sample_train.tsv --test data\sample_test.tsv --images data\images --labels 6-way --limit 6000 --batch-size 16
```

Training keeps CLIP frozen and fits the classifier over embeddings. It does not fine-tune the full transformer, which keeps GPU memory use practical on a 6 GB RTX 4050. If memory is tight, reduce `--batch-size` to 8 or 4. Training writes `artifacts/clip_fakeddit_classifier.joblib` and `artifacts/evaluation.json`.

Fakeddit six-way labels are mapped for this course task as follows: 0 True → Real; 2 False Connection and 4 Manipulated → Fake; 1 Satire, 3 Imposter, and 5 Misleading → Possibly misleading / uncertain. This is a project policy; Fakeddit labels are platform-specific and do not define universal truth.

## Original Fakeddit evaluation

The test split is not used to select the confidence threshold. Training examples are split again for probability calibration and threshold selection. Final test metrics are saved in `artifacts/evaluation.json`; run training to regenerate them for your own subset. Report accuracy, macro-F1, per-class results, and the confusion matrix as a dataset-specific measurement, not general-world accuracy.

## Reproduce the showcase checks

On the 32 previously reported AVeriTeC cases the updated app matched 20 labels, up from 15. The three previous false alarms on ambiguous claims were resolved. Of 16 claims labeled true or false, four received the correct verdict and twelve received Uncertain.

On a separate randomly sampled set of 40 AVeriTeC claims (seed 78551, excluding all earlier samples and exact matches to the local AVeriTeC training archive), the app gave four correct verdicts and abstained 36 times. The previous release gave five correct verdicts, one incorrect verdict and 34 abstentions. Both releases checked identical claim text without a supplied article, date, original fact-check URL or expected label. Some claims may already have published reviews in the independent publisher index. Broad automatic news coverage is poor. Reproduce the revised run with `python evaluate_showcase.py --release`. The before-and-after predictions and cited passages are under `data/showcase_eval`. Live sources may change.

Six development captions scored 5/6. On twelve previously used Oxford pet test photos the updated detector identified eleven labeled animals, up from nine. On 24 separately sampled Oxford test photos it identified all 24, up from 21. Those 24 captions all described the labeled species. This measures object recognition and does not evaluate false captions or relationships.

To rebuild the assets, run `python download_visual_models.py`, `python build_reference_index.py` and `python publisher_index.py data/publisher_snapshot.zip`. The publisher-index builder verifies the archived release checksum. Existing CLIP and DeBERTa models load from the Hugging Face cache. Install matching PyTorch and TorchVision builds for your GPU.

Publisher metadata is adapted from the [Martino Mensio ClaimReview snapshot of November 25, 2026](https://github.com/MartinoMensio/claimreview-data/releases/tag/2026_11_25), distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The derived index preserves original claims, ratings, publication dates and publisher URLs. Other evidence and evaluation sources include [AVeriTeC](https://fever.ai/dataset/averitec.html), the [Wikimedia snapshot](https://huggingface.co/datasets/wikimedia/wikipedia), [Wikidata](https://www.wikidata.org/), and [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).

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
