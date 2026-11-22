# CrossCheck AI: Multimodal Fake News Screening

## Abstract

CrossCheck AI screens a social-media claim together with its associated image. It uses pretrained CLIP ViT-B/32 to encode text and image, then applies a supervised classifier trained on Fakeddit examples. It returns **Fake**, **Real**, or **Possibly misleading / uncertain**, shows class scores and semantic similarity, and can retrieve links for a person to inspect. The result is an educational model signal, not proof of factual truth.

## Problem and approach

A post can use a genuine image with an unrelated or misleading caption, or a relevant image that is old or taken out of context. Text-image alignment is one useful clue, but it cannot establish provenance, date, manipulation, or truth. This project combines that clue with a classifier trained on labeled paired examples and keeps an explicit uncertain output for low-confidence predictions.

## System

1. The Flask interface accepts claim text and its associated image.
2. A frozen pretrained CLIP dual encoder creates normalized text and image embeddings.
3. The feature vector concatenates the text embedding, image embedding, their elementwise product, and their cosine similarity.
4. A class-balanced logistic regression classifier predicts one of three project-mapped categories. Sigmoid calibration is fitted using training data only.
5. A confidence threshold is selected on a held-out slice of the training partition. Predictions below that threshold are routed to **Possibly misleading / uncertain**.
6. Optional evidence search tries DDGS, Google News RSS, and Wikipedia. If providers fail or return no links, analysis still completes and the interface explains that search is unavailable.

## Dataset and label mapping

Fakeddit was selected because it includes social-post text, labels, and associated images. The working subset has **5,686 paired train examples** and **1,432 paired held-out test examples** after missing/unavailable images were excluded. It retains Fakeddit's `general_train` and `general_test` partitions.

The project maps six-way labels as follows: 0 True → Real; 2 False Connection and 4 Manipulated Content → Fake; 1 Satire, 3 Imposter, and 5 Misleading Content → Possibly misleading / uncertain. This is a task-specific mapping, not a universal definition of truth. The dataset is Reddit-derived and its labels can be noisy or platform-specific.

## Held-out evaluation

| Measure | Result |
|---|---:|
| Test pairs | 1,432 |
| Accuracy | 84.6% |
| Macro-F1 | 0.799 |
| Confidence threshold | 37.5% |

Per-class test results:

| Project class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Fake | 0.871 | 0.892 | 0.882 | 706 |
| Real | 0.844 | 0.850 | 0.847 | 566 |
| Possibly misleading / uncertain | 0.719 | 0.625 | 0.669 | 160 |

The uncertain class remains the weakest class in this run. These scores describe only this held-out Fakeddit subset; they should not be presented as expected accuracy on current real-world news.

## Hardware and implementation

The model uses `openai/clip-vit-base-patch32`, PyTorch, Transformers, scikit-learn, and Flask. The CLIP encoder remains frozen, so only a lightweight classifier is fit. The target laptop has an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. The web interface uses HTML, CSS, and JavaScript and runs locally at `http://127.0.0.1:8501`.

## Limitations and responsible use

- The classifier learns patterns in Fakeddit and inherits its label noise, distribution, and category definitions.
- CLIP similarity represents semantic alignment, not authenticity, chronology, source, or factual correctness.
- A calibrated class score is still a model score, not the probability that a real-world claim is true.
- Search results are investigation leads. Provider errors or snippets do not determine the model prediction.
- Human review and trusted primary sources remain necessary. This educational prototype is not intended for high-stakes decisions or labeling individuals.

## References

1. Nakamura, K., Levy, S., & Wang, W. Y. (2020). *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*. LREC. https://aclanthology.org/2020.lrec-1.755/
2. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. https://arxiv.org/abs/2103.00020
3. OpenAI, [CLIP ViT-B/32 model card](https://huggingface.co/openai/clip-vit-base-patch32).
4. [Fakeddit repository](https://github.com/entitize/Fakeddit) and [dataset mirror](https://huggingface.co/datasets/AdoCleanCode/Fakeddit).
