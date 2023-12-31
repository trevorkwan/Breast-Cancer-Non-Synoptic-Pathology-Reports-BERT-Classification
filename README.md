## Breast Cancer Non-Synoptic Pathology Reports BERT Classification

**Goal**: Extract Cancer Fields of Interest (FOI) from Non-Synoptic Reports.

A non-synoptic report is a free-text report with little-to-no structure, but may contain information for cancer fields of interest.

For example, given a report, one cancer field of interest is Invasive Carcinoma. We classify the presence of Invasive Carcinoma as "Present", "Absent", or in the case where the report doesn't contain information on Invasive Carcinoma "No Mention".

### Methodology
Step 1: Extract the relevant span of text from the entire report. (Question Answering)  
Step 2: Classify the extracted span of text into a class. (Text Classification)  

#### Data Loading, Cleaning, and Exploratory Data Analysis (EDA)
We [load and clean](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/load_and_clean_data.py) synoptic and non-synoptic report data, removing irrelevant text/labels and accounting for "No Mention" data. We do [EDA](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/EDA.py) on the cleaned data to look for data imbalance and abnormalities. Data cleaning and EDA are an iterative process as we redefine project goals with healthcare stakeholders.

#### Question Answering (QA) Training
We fine-tune two pretrained clinical models on squadv2: [pubmed_bert_squadv2](https://huggingface.co/franklu/pubmed_bert_squadv2) and [bluebert_pubmed_mimic_squadv2](https://huggingface.co/trevorkwan/bluebert_pubmed_mimic_uncased_squadv2), then further [fine-tune](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/train_v2.py) them on our own cancer pathology report data. We fine-tune over 30 models to optimize hyperparameters, balancing model performance with overfitting.

#### Quesiton Answering (QA) Evaluation and Inference
We [evaluate](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/eval_v3.py) the trained QA models and extract their predictions. Taking the predictions, we compute the [optimal QA score threshold](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/error_analysis_qa_score.ipynb) and classify any prediction below this threshold as "No Mention". We [infer](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/inference_v2.py) model prediction performance on f1-score, Precision, and Recall metrics.

#### Text Classification Training and Evaluation
We fine-tune [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on gold standard manual report annotations and [classify](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/classify_v3.py) QA model extracted text predictions. We evaluate classification accuracy within and across cancer fields of interest.

#### Other Classification Methods
We explore other classification methods with varying performances for project thoroughness, including [simple regex](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/regex_classification.ipynb), [weighted keywords](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/weighted_keyword_analysis.ipynb), and [naive bayes](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/naive_bayes_classifier.ipynb).

#### Exploratory Methods for QA Optimization
We explore methods to optimize QA model performance, including [downsampling](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/get_sub_11_and_downsampled.ipynb), removing metadata, and adding synoptic data.

### Results
We automate classification of non-synoptic reports with near-human-level accuracy, greater than 90%, in multiple cancer fields of interest.
