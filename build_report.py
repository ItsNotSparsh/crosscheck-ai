from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
import json

OUT=Path(__file__).parent/'CrossCheck_AI_Final_Project_Report.docx'
EVAL=Path(__file__).parent/'artifacts'/'evaluation.json'
TRAIN_TSV=Path(__file__).parent/'data'/'sample_train.tsv'
TEST_TSV=Path(__file__).parent/'data'/'sample_test.tsv'
METRICS=json.loads(EVAL.read_text(encoding='utf-8')) if EVAL.exists() else {}
def row_count(path):
    return max(0, sum(1 for _ in path.open(encoding='utf-8'))-1) if path.exists() else 0
TRAIN_COUNT=row_count(TRAIN_TSV); TEST_COUNT=row_count(TEST_TSV)
ACCURACY=f"{METRICS.get('accuracy',0):.1%}" if METRICS else 'Run training to generate'
MACRO_F1=f"{METRICS.get('macro_f1',0):.3f}" if METRICS else 'Run training to generate'
THRESHOLD=f"{METRICS.get('confidence_threshold',0):.0%}" if METRICS else 'validation selected'
d=Document(); sec=d.sections[0]; sec.top_margin=Inches(.72); sec.bottom_margin=Inches(.72); sec.left_margin=Inches(.82); sec.right_margin=Inches(.82)
styles=d.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(10)
styles['Normal'].paragraph_format.space_after=Pt(5)
title_ppr=styles['Title']._element.get_or_add_pPr()
title_border=title_ppr.find(qn('w:pBdr'))
if title_border is not None: title_ppr.remove(title_border)
for name,size,color in [('Title',25,'153E75'),('Heading 1',16,'153E75'),('Heading 2',12,'285EA8')]:
    s=styles[name]; s.font.name='Aptos Display'; s.font.size=Pt(size); s.font.color.rgb=RGBColor.from_string(color)

def shade(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def table(headers, rows, widths=None):
    t=d.add_table(rows=1, cols=len(headers)); t.style='Light Shading Accent 1'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=h; shade(c,'153E75')
        for r in c.paragraphs[0].runs: r.font.bold=True; r.font.color.rgb=RGBColor(255,255,255)
    for row in rows:
        cells=t.add_row().cells
        for i,val in enumerate(row): cells[i].text=val; cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    d.add_paragraph()
    return t
def bullet(text): d.add_paragraph(text,style='List Bullet')

p=d.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('CrossCheck AI Multimodal Fake News Screening Project Report'); r.bold=True; r.font.size=Pt(24); r.font.color.rgb=RGBColor(21,62,117)
p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Multimodal Fake News Screening with Vision-Language Models'); r.font.size=Pt(15); r.font.color.rgb=RGBColor(70,82,100)
p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Applications of AI | Final Project Report | November 2026')
d.add_paragraph('Project team: Prajjwal Sharma (S24CSEU1391), Vikalp Chaudhary (S24CSEU0148), Sparsh Kumar (S24CSEU0428), Ajitej Singh (S24CSEU1413)',style='Subtitle')
d.add_heading('Abstract',1)
d.add_paragraph('CrossCheck AI is a student decision-support prototype that screens a social-media post by analyzing its claim text and associated image together. It uses the pretrained CLIP ViT-B/32 vision-language transformer to create aligned text and image representations, then applies a lightweight classifier trained on paired examples from Fakeddit. The interface returns Fake, Real, or Possibly misleading / uncertain, reports a text-image compatibility cue, can retrieve public web results for follow-up, and explains the visible signals. The result is a screening alert and is not proof of factual truth.')
d.add_heading('1. Problem and project aim',1)
d.add_paragraph('A misleading post can pair a plausible claim with a genuine but unrelated, old, manipulated, or misleadingly captioned image. A text-only review may miss that relationship. This project applies a pretrained vision-language model to evaluate the post and image jointly and routes ambiguous cases to a human reviewer.')
d.add_paragraph('Aim: develop a working, explainable multimodal screening application that predicts Fake, Real, or Possibly misleading / uncertain and helps the user inspect cross-modal cues and external evidence.')
d.add_heading('2. Objectives addressed',1)
for x in ['Detect potential text-image inconsistencies with CLIP embedding similarity.','Retrieve external search results as evidence leads when a user chooses to investigate a claim.','Explain the prediction using the predicted class, classifier confidence, compatibility cue, and linked evidence leads.','Evaluate on a held-out Fakeddit split and report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix.'] : bullet(x)
d.add_heading('3. Dataset selection',1)
d.add_paragraph(f'Fakeddit is selected because the project requires paired social-post text and images. We use the 73.9 MB AdoCleanCode/Fakeddit Hugging Face parquet mirror, which includes general_train and general_test flags, text, labels, and image URLs. The full image archive is over 100 GB, so a limited sample was downloaded from those URLs. After unavailable or missing images were excluded, the current working subset contains {TRAIN_COUNT:,} training pairs and {TEST_COUNT:,} held-out test pairs. The encoder remained frozen and the classifier was trained on fused CLIP representations.')
d.add_paragraph('Project mapping for the six-way labels: 0 True → Real; 1 Satire → Possibly misleading / uncertain; 2 False Connection → Fake; 3 Imposter → Possibly misleading / uncertain; 4 Manipulated Content → Fake; 5 Misleading Content → Possibly misleading / uncertain. This category mapping is a project policy. Dataset labels are platform-specific and are not a universal definition of truth.')
d.add_heading('4. System design',1)
table(['Stage','Implementation'],[
('Input','User provides post text and one associated image in the Flask web interface.'),
('Preprocessing','CLIP processor tokenizes text and resizes/normalizes the image.'),
('VLM representations','Frozen CLIP ViT-B/32 creates normalized text and image embeddings.'),
('Cross-modal feature','Cosine similarity measures semantic compatibility; fused feature combines text, image, elementwise interaction, and similarity.'),
('Classification','Class-balanced logistic regression trained on frozen multimodal features predicts the three mapped classes.'),
('Uncertainty','Sigmoid-calibrated class scores are used with a confidence threshold selected from a training-only validation slice. Below-threshold results route to Possibly misleading / uncertain.'),
('Evidence and explanation','Optional web search surfaces source links/snippets as leads; the app displays the model signals and caveats.')])
d.add_heading('5. Model and implementation',1)
d.add_paragraph('The encoder is openai/clip-vit-base-patch32, a pretrained dual-encoder vision-language transformer. Its weights remain frozen to keep compute practical on an RTX 4050 Laptop GPU with 6 GB VRAM. A class-balanced logistic regression classifier is fit on the resulting embeddings rather than fine-tuning the full transformer. Sigmoid probability calibration and the abstention threshold use training data only. PyTorch runs inference on CUDA when available; Flask serves the responsive HTML, CSS, and JavaScript interface.')
d.add_paragraph('The three-way decision does not treat a low similarity score as evidence that a claim is false. Similarity only indicates whether image content and claim wording appear semantically related. A matching image can still be old or miscaptioned, and a valid claim can use an illustrative image.')
d.add_heading('6. Data preparation and run instructions',1)
d.add_paragraph('The prepared parquet mirror and trained model are included in the project workspace. To reproduce the sample data and run the application, use:')
for cmd in ['python prepare_data.py data\\fakeddit_train.parquet data\\sample_train.tsv data\\images --split train --limit 6000','python prepare_data.py data\\fakeddit_train.parquet data\\sample_test.tsv data\\images --split test --limit 1500','python train.py --train data\\sample_train.tsv --test data\\sample_test.tsv --images data\\images --labels 6-way --limit 6000 --batch-size 16','python app.py  # then open http://127.0.0.1:8501']:
    p=d.add_paragraph(); p.paragraph_format.left_indent=Inches(.25); run=p.add_run(cmd); run.font.name='Consolas'; run.font.size=Pt(9)
d.add_paragraph('The CLIP model is downloaded to the local Hugging Face cache. Training produces artifacts/clip_fakeddit_classifier.joblib and artifacts/evaluation.json. Optional evidence search queries DDGS, Google News RSS, and Wikipedia in parallel under a short overall deadline. Search-provider errors do not prevent the classifier from returning its analysis.')
d.add_heading('7. Evaluation plan',1)
d.add_paragraph(f'We retained the dataset mirror’s general_train and general_test partitions. On {TEST_COUNT:,} paired held-out examples, this run achieved {ACCURACY} accuracy and {MACRO_F1} macro-F1. The threshold selected using the training-only validation slice was {THRESHOLD}. These are dataset-specific measurements, not a claim of broad real-world performance. Review per-class results in artifacts/evaluation.json because class-wise performance may differ.')
table(['Measure','Observed result','Purpose'],[('Accuracy',ACCURACY,'Overall share of correct predictions.'),('Macro-F1',MACRO_F1,'Balances class performance when categories are imbalanced.'),('Test pairs',f'{TEST_COUNT:,}','Paired examples retained in the held-out split.'),('Training pairs',f'{TRAIN_COUNT:,}','Paired examples retained from the training split.'),('Abstention threshold',THRESHOLD,'Selected using only the training validation slice.')])
report=METRICS.get('report',{})
class_rows=[]
for label,name in [('0','Fake'),('1','Real'),('2','Possibly misleading / uncertain')]:
    values=report.get(label,{})
    if values:
        class_rows.append((name,f"{values.get('precision',0):.3f}",f"{values.get('recall',0):.3f}",f"{values.get('f1-score',0):.3f}",f"{int(values.get('support',0)):,}"))
if class_rows: table(['Class','Precision','Recall','F1','Support'],class_rows)
d.add_heading('8. Limitations and responsible use',1)
for x in ['Fakeddit is derived from Reddit and can contain noisy, dated, or platform-specific labels.','CLIP measures semantic compatibility; it does not establish image provenance, recency, manipulation, or claim truth.','Web search snippets may be incomplete or unreliable. Users should open and assess primary or authoritative sources.','The uncertain class is a conservative system output, not a ground-truth claim that content is partly true.','The prototype is for educational triage and human review. It should not be used for punitive moderation, high-stakes decisions, or labeling individuals.'] : bullet(x)
d.add_heading('9. Conclusion',1)
d.add_paragraph('CrossCheck AI delivers a practical multimodal screening workflow using a pretrained vision-language transformer, a resource-efficient classifier, an uncertainty route, visible text-image compatibility, and optional evidence leads. Fakeddit is appropriate for training this paired-input prototype, provided its label mapping, missing images, and dataset limitations are documented. Its output supports investigation; human verification remains essential.')
d.add_heading('References',1)
for ref in ['Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection. LREC. https://aclanthology.org/2020.lrec-1.755/','Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML. https://arxiv.org/abs/2103.00020','OpenAI. CLIP ViT-B/32 model card. https://huggingface.co/openai/clip-vit-base-patch32','Fakeddit project repository and dataset use information. https://github.com/entitize/Fakeddit','AdoCleanCode/Fakeddit dataset mirror. https://huggingface.co/datasets/AdoCleanCode/Fakeddit'] : d.add_paragraph(ref,style='List Number')
d.save(OUT)
print(OUT)
