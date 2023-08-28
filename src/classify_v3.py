import pandas as pd
import os
import torch

from transformers import AutoTokenizer,  AutoModelForQuestionAnswering
from transformers import TrainingArguments, Trainer, default_data_collator
from datasets import Dataset

import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pandas as pd
import uuid
import time
import torch

import pandas as pd
import numpy as np
import re
import os

from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer, set_seed, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

"""
Takes the a label_value and answer from the training data and trains a classifier.
Takes a qa_answer from the validation data (prediction csvs from qa model) and classifies them.
Computes the accuracies across FOIs comparing actual label_value and classification.
"""

# Set the seed
set_seed(42)

# Define the arguments
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 3
weight_decay = 0.01
max_seq_length = 480 # The maximum length of a feature (question and context)
train_text_column_name = "answer"
val_text_column_name = "qa_answer"
label_column_name = "label_value_num"
max_train_samples = None
max_eval_samples = None
qa_score_threshold = "0"

oversample_percentage = '0%'
comment = 'qa score threshold predictions of 0.25. classifier trained on gold standard answers from 40% train data subset by FOI. evaluated on 30% validation data from the output of the QA model, which are predictions. can we classify the span of text predictions in classes?'

model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT" # for classifier
model_name = model_checkpoint.split("/")[-1]
model_version = "_v2"

original_model_checkpoint = 'franklu/pubmed_bert_squadv2' # for qa
original_model_name = original_model_checkpoint.split("/")[-1]
original_model_dir = "../results/trained\\"
original_model_signature = '_19827_v2\\' # for qa
version = 'v6'

desired_fois = ["DCIS Margins", "ER Status", "Extranodal Extension", "HER2 Status", "Insitu Component", "Invasive Carcinoma", "Invasive Carcinoma Margins", "Lymphovascular Invasion", "Necrosis", "PR Status", "Tumour Focality"]
# desired_fois = ["Invasive Carcinoma", "Insitu Component"]

# Initialize a list to keep track of the accuracy for each FOI
accuracy_summary_list = []

# Initialize a dictionary to keep track of the label counts for each FOI
label_counts_summary = {}

for foi in desired_fois:
    print(f"Training model for Field of Interest: {foi}")

    # load the data
    train_df = pd.read_csv(f"../data/clean/non_synoptic/complete/train_FOI/train_{foi}.csv")
    val_df = pd.read_csv(original_model_dir + original_model_name + original_model_signature + "eval\\" + version + "\\" + foi + "\\" + 'predictions_' + foi + '.csv')

    # in val pred csvs, add No Mention
    val_df['label_value'].fillna('No Mention', inplace=True)
    val_df['label_value'].replace('', 'No Mention', inplace=True)

    # numerically code the outcome variable
    le = LabelEncoder()
    # Fit the LabelEncoder with all possible categories
    le.fit(pd.concat([train_df['label_value'], val_df['label_value']]))
    train_df['label_value_num'] = le.transform(train_df['label_value'])
    val_df['label_value_num'] = le.transform(val_df['label_value'])  # Use transform() not fit_transform() to ensure consistent encoding

    # Get the count of each class in the training data
    label_counts = train_df['label_value'].value_counts()

    # Save the label counts for the current FOI in the label_counts_summary dictionary
    label_counts_summary[foi] = label_counts

    # get the number of labels
    num_labels = pd.concat([train_df['label_value'], val_df['label_value']]).nunique()
    print("Num labels:", num_labels)
    # num_labels = len(set(combined_df['label_value_num']))  # Get number of labels from training data

    # Define the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels) 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # use cuda if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Using CUDA")
        model = model.to("cuda:0")
        per_device_train_batch_size = 8  # in case CUDA runs out of memory
        per_device_eval_batch_size = 8  # in case CUDA runs out of memory

    # checks if the tokenizer is a fast tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # checks max sequence length not longer than tokenizer max length
    if max_seq_length > tokenizer.model_max_length:
        print("The max_seq_length passed is larger than the tokenizer max length. Using tokenizer max length instead.")
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    if max_train_samples is not None:
        # subset for testing purposes
        max_train_samples = min(len(train_df), max_train_samples)
        train_df = train_df.iloc[:max_train_samples]

    if max_eval_samples is not None:
        # subset for testing purposes
        max_eval_samples = min(len(val_df), max_eval_samples)
        val_df = val_df.iloc[:max_eval_samples]

    train_nrows = len(train_df)
    print("Train sample size:", train_nrows)
    val_nrows = len(val_df)
    print("Val sample size:", val_nrows)

    # Define model signature
    model_signature = '_' + str(train_nrows) + model_version

    # Create a list of datasets
    def preprocess_function_train(examples):
        # Tokenize the text
        tokenized = tokenizer(examples[train_text_column_name], truncation=True, max_length=512, padding='max_length')
        # Include the labels
        tokenized['labels'] = list(examples[label_column_name])
        return tokenized
    
    # Create a list of datasets
    def preprocess_function_val(examples):
        # Tokenize the text
        tokenized = tokenizer(examples[val_text_column_name], truncation=True, max_length=512, padding='max_length')
        # Include the labels
        tokenized['labels'] = list(examples[label_column_name])
        return tokenized

    # Fill NaN values with ""
    train_df[train_text_column_name].fillna("", inplace=True)
    val_df[val_text_column_name].fillna("", inplace=True)

    train_dataset = Dataset.from_pandas(train_df[[label_column_name, train_text_column_name]]).map(preprocess_function_train, batched=True) # For training model only keep label and text
    val_dataset = Dataset.from_pandas(val_df[[label_column_name, val_text_column_name]]).map(preprocess_function_val, batched=True)

    data_collator = default_data_collator

    output_dir = os.path.join(original_model_dir, original_model_name + original_model_signature, "eval", version, foi, "trained")

    args = TrainingArguments(
        output_dir = os.path.join(output_dir, model_name + model_signature),
        evaluation_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset= train_dataset,
        eval_dataset= val_dataset,
        data_collator=data_collator,
        tokenizer= tokenizer,
    )

    trainer.train()

    # Once the model is done training, save the result
    trainer.save_model(os.path.join(output_dir, model_name + model_signature))

    # Define the model card content as a string
    model_card_content = f'''Base Model: {model_checkpoint}
    Model Name: {model_name}
    Model Version (Signature): {model_signature}
    Task: Classification
    Number of Train Samples: {train_nrows}
    Numnber of Val Samples: {val_nrows}
    Oversample Percentage: {oversample_percentage}
    Learning Rate: {learning_rate}
    Number of Train Epochs: {num_train_epochs}
    Weight Decay: {weight_decay}
    Max Seq Length: {max_seq_length}
    Train Batch Size: {per_device_train_batch_size}
    Validation Batch Size: {per_device_eval_batch_size}
    QA Score Threshold: {qa_score_threshold}
    Comment: {comment}
    '''

    # Define the file path for the model card text file
    model_card_file_path = os.path.join(output_dir, model_name + model_signature) + r'\model_card.txt'

    # Write the model card content to the text file
    with open(model_card_file_path, 'w', encoding='utf-8') as file:
        file.write(model_card_content)

    print(f"Model card has been saved to: {model_card_file_path}")

    #### Inference on trained model ####

    # Import model, tokenizer (if not already loaded)
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir, model_name + model_signature), num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_len = 512)

    class_pipe = pipeline('text-classification', model= model, tokenizer= tokenizer)

    def class_model_inference(text, class_pipe = class_pipe):
        query = class_pipe(text, truncation = True)
        label = query[0]['label']
        # Extract the integer from the label string
        label = int(label.split('_')[1])
        return([label, query[0]['score']])

    val_df[['classifier_predicted_label', 'classifier_score']] = val_df.apply(lambda x: pd.Series(class_model_inference(text = x[val_text_column_name])), axis = 1)

    test_pred_df = val_df[[label_column_name, val_text_column_name, 'label_value', 'classifier_predicted_label', 'classifier_score']]

    # Save the DataFrame to a CSV file
    new_output_dir = os.path.join(output_dir, model_name + model_signature, "eval")
    # Create the output directory (foi folder) if it doesn't exist
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    test_pred_df.to_csv(os.path.join(new_output_dir,'classifier_predictions_' + foi + '.csv'), index=False)

    # Unique classes
    classes = test_pred_df['label_value'].unique()

    # Prepare a dictionary to store accuracy results for the current FOI
    accuracy_dict = {}

    # Overall accuracy
    overall_accuracy = accuracy_score(test_pred_df['label_value_num'], test_pred_df['classifier_predicted_label'])
    # accuracy_dict['Overall'] = overall_accuracy
    print(f"Overall Accuracy for '{foi}': {overall_accuracy}")

    # Append the overall accuracy to the summary list
    accuracy_summary_list.append({'type': 'overall_accuracy', 'foi': foi, 'label_value': None, 'accuracy': overall_accuracy})

    # Unique classes
    classes = test_pred_df['label_value'].unique()

    # Prepare a dictionary to store accuracy results for the current FOI
    accuracy_dict = {'FOI': foi, 'Overall': overall_accuracy}

    # Accuracy for each class
    for class_label in classes:
        class_accuracy = accuracy_score(test_pred_df[test_pred_df['label_value'] == class_label]['label_value_num'], 
                                        test_pred_df[test_pred_df['label_value'] == class_label]['classifier_predicted_label'])
        accuracy_dict[class_label] = class_accuracy
        print(f"Accuracy for '{class_label}' in '{foi}': {class_accuracy}")

        accuracy_summary_list.append({'type': 'label_value_accuracy', 'foi': foi, 'label_value': class_label, 'accuracy': class_accuracy})

    # Convert the dictionary to a dataframe
    accuracy_df = pd.DataFrame(list(accuracy_dict.items()), columns=['Class', 'Accuracy'])

    # Save the dataframe to a CSV file
    accuracy_df.to_csv(os.path.join(new_output_dir,'classifier_accuracy_' + foi + '.csv'), index=False)

# Convert the accuracy_summary_list into a DataFrame
accuracy_summary_df = pd.DataFrame(accuracy_summary_list)

# Calculate the mean accuracy for "No Mention" across all FOIs
no_mention_mean_accuracy = accuracy_summary_df[accuracy_summary_df['label_value'] == 'No Mention']['accuracy'].mean()
# Create a DataFrame for the overall "No Mention" mean accuracy
overall_no_mention_accuracy_df = pd.DataFrame({
    'type': ['overall_accuracy'],
    'foi': ['No Mention'],
    'label_value': [None],
    'accuracy': [no_mention_mean_accuracy]
})
# Concatenate the overall "No Mention" mean accuracy with the combined_accuracies_df
accuracy_summary_df = pd.concat([accuracy_summary_df, overall_no_mention_accuracy_df], ignore_index=True)
accuracy_summary_df = accuracy_summary_df.sort_values(by='type', ascending=True) # sort by type column
accuracy_summary_df = accuracy_summary_df.reset_index(drop=True)

# Save the accuracy_summary DataFrame as a CSV file
accuracy_summary_df.to_csv(original_model_dir + original_model_name + original_model_signature + "eval\\" + version + "\\" + "average_classifier_metrics.csv", index=False)

# Convert the label_counts_summary dictionary into a DataFrame
label_counts_summary_df = pd.DataFrame(list(label_counts_summary.items()), columns=['FOI', 'Label Counts'])

# Save the label_counts_summary DataFrame as a CSV file
label_counts_summary_df.to_csv(original_model_dir + original_model_name + original_model_signature + "eval\\" + version + "\\" + "train_FOI_label_counts.csv", index=False)
