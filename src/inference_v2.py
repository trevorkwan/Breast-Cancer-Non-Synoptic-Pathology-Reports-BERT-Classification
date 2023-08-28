import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from difflib import SequenceMatcher
import os
from transformers import set_seed
import csv 

# Set the seed
set_seed(42)

# Define the arguments
model_checkpoint = 'franklu/pubmed_bert_squadv2'
model_name = model_checkpoint.split("/")[-1]
model_dir = "../results/trained\\"
model_signature = '_19827_v3\\'
version = 'v2'

desired_fois = ["DCIS Margins", "ER Status", "Extranodal Extension", "HER2 Status", "Insitu Component", "Invasive Carcinoma", "Invasive Carcinoma Margins", "Lymphovascular Invasion", "Necrosis", "PR Status", "Tumour Focality"]
input_dir = os.path.join(model_dir, model_name + model_signature, "eval", version)

# df = pd.read_csv(model_dir + model_name + model_signature + "eval\\" + foi + "\\" + 'predictions_' + foi + '.csv')

def compute_f1_precision_recall(gold_answer, predicted_answer):
    """
    Gets F1 metric
    :param gold_answer: gold standard answer str
    :param predicted_answer: model predicted answer str
    :return: F1 score
    """
    gold_tokens = str(gold_answer).strip("\n\t _-,.*!?\\/\"'").lower().split() # each token is always a word
    pred_tokens = str(predicted_answer).strip("\n\t _-,.*!?\\/\"'").lower().split() # each token is always a word

    num_shared_tokens = len(set(gold_tokens) & set(pred_tokens)) # number of shared words

    # Calculate precision
    precision = num_shared_tokens / len(pred_tokens) if len(pred_tokens) > 0 else 0.0 # number of shared words out of all predicted words

    # Calculate recall
    recall = num_shared_tokens / len(gold_tokens) if len(gold_tokens) > 0 else 0.0 # number of shared words out of all actual words

    # Calculate f1 score
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0 # the balance of both

    return f1, precision, recall

def get_f1_precision_recall_scores(df):
    """
    Creates metric dataframe and visualizations for given dataframe
    :param df: pandas df
    :return: df
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for index, row in df.iterrows():
        gold_answer = row['answer']
        predicted_answer = row['qa_answer']

        f1, precision, recall = compute_f1_precision_recall(gold_answer, predicted_answer)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        metric_df = pd.DataFrame({
        "f1_score": f1_scores,
        "precision": precision_scores,
        "recall": recall_scores
    })

    return metric_df

def get_average_scores(metric_df):
    """
    Computes the average for each column in the provided DataFrame
    :param metric_df: pandas DataFrame containing metrics
    :return: pandas Series containing the average of each column in metric_df
    """
    avg_scores = metric_df.mean()

    # Convert the series to a DataFrame and transpose it for saving to CSV
    avg_scores_df = avg_scores.to_frame().T

    return avg_scores_df

def process_foi(foi):
    df = pd.read_csv(input_dir + "/" + foi + "/" + 'predictions_' + foi + '.csv')

    metric_df = get_f1_precision_recall_scores(df)

    avg_metrics_df = get_average_scores(metric_df)

    # Append average metrics to the bottom of metric_df
    avg_metrics_df.index = ['average']
    metric_df = pd.concat([metric_df, avg_metrics_df])

    # Save the DataFrame to a CSV file
    output_dir = os.path.join(input_dir, foi)

    # Create the output directory (foi folder) if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_df.to_csv(os.path.join(output_dir, 'metrics_' + foi + '.csv'))

# Apply process_foi to each desired FOI
for foi in desired_fois:
    process_foi(foi)

# Compute overall metrics
total_f1_score_avg = 0
total_precision_avg = 0
total_recall_avg = 0

f1_scores_avg_dict = {}
precision_avg_dict = {}
recall_avg_dict = {}

for foi in desired_fois:
    filepath = os.path.join(input_dir, foi, f'metrics_{foi}.csv')

    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[''] == 'average':
                f1_score_avg = round(float(row['f1_score']), 3)
                precision_avg = round(float(row['precision']), 3)
                recall_avg = round(float(row['recall']), 3)
                
                total_f1_score_avg += f1_score_avg
                total_precision_avg += precision_avg
                total_recall_avg += recall_avg
                
                f1_scores_avg_dict[foi] = f1_score_avg
                precision_avg_dict[foi] = precision_avg
                recall_avg_dict[foi] = recall_avg

# Calculate the averages across all CSVs
avg_f1_score = total_f1_score_avg / len(desired_fois)
avg_precision = total_precision_avg / len(desired_fois)
avg_recall = total_recall_avg / len(desired_fois)

# Create a DataFrame for the average metrics
average_metrics = pd.DataFrame({
    "average_f1_score": [avg_f1_score],
    "average_precision": [avg_precision],
    "average_recall": [avg_recall],
    "average_f1_scores_dict": [f1_scores_avg_dict],
    "average_precision_dict": [precision_avg_dict],
    "average_recall_dict": [recall_avg_dict]
})

# Save the average metrics to a CSV file
average_metrics.to_csv(os.path.join(input_dir, 'average_qa_metrics.csv'), index=False)