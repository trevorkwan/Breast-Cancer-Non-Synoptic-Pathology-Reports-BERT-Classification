## Breast Cancer Non-Synoptic Pathology Reports BERT Classification

**Goal**: Extract Cancer Fields of Interest (FOI) from Non-Synoptic Reports.

A non-synoptic report is a free-text report with little-to-no structure, but may contain information for cancer fields of interest.

For example, given a report, one cancer field of interest is Invasive Carcinoma. We classify the presence of Invasive Carcinoma as "Present", "Absent", or in the case where the report doesn't contain information on Invasive Carcinoma "No Mention".

### Methodology
Step 1: Extract the relevant span of text from the entire report. (Question Answering)  
Step 2: Classify the extracted span of text into a class. (Text Classification)  

#### Data Loading, Cleaning, and Exploratory Data Analysis (EDA)
We [load and clean](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/load_and_clean_data.py) synoptic and non-synoptic report data, removing irrelevant text/labels and accounting for missing data. We do [EDA](https://github.com/trevorkwan/Breast-Cancer-Non-Synoptic-Pathology-Reports-BERT-Classification/blob/main/src/EDA.py) on the cleaned data to look for data imbalance and abnormalities. Data cleaning and EDA were an iterative process as we redefine project goals with healthcare stakeholders.

#### Question Answering (QA) Training
- Choosing the QA pretrained model to use.

#### Quesiton Answering (QA) Evaluation and Inference

Take the predictions to get optimal QA score based on Yes/No Mentions. Re-evaluate on optimal QA score threshold.
f1-score, Precision, Recall

#### Text Classification Training and Evaluation

#### Other Classification Methods

#### Exploratory Methods for Optimization (QA Downsampling, Removing Metadata, Adding Synoptic Data)


- ipynbs of interest: error_analysis_qa_score, get_sub_11_and_downsampled, naive_bayes_classifier, regex_classification, weighted_keyword_analysis
