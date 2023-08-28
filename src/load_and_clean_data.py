# import libraries

import os
import re
import json
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import set_seed

# Set the seed
set_seed(42)

# Define the Arguments
train_split_perc = 0.3
val_split_perc = 0.3
test_split_perc = 0.4

# define the parents folder, reports, and unique_id
parent = r"\\srvnetapp02.vch.ca\bcca\docs\EVERYONE\Cancer Registry Requests\Secure_File_Location\Cancer_NLP_Project\emarc_data\Annotated Reports\Non-Synoptic - NEWEST VERSION\\"
folders = ['Annah Non-Synoptic 1.0', 'Jenna Non-Synoptic 1.0', 'Revised Non-Synoptic 1.0 - Christy', 'Revised Non-Synoptic 2.0 - Christy', 'Revised Non-Synoptic 3.0 - Christy', 'Revised Non-Synoptic 4.0 - Christy']
reports = []
unique_id = 0

# load all non-synoptic reports (each line is a report)
for folder in folders:
    os.chdir(parent + "\\" + folder)
    with open("admin.jsonl") as f:
        for line in f: # each line is a report with an id, text, and labels
            unique_id = unique_id + 1 # each unique_id is a report
            json_report = json.loads(line)
            json_report['report_id'] = unique_id
            json_report['folder'] = re.findall('Christy|Annah|Jenna', folder).pop()
            reports.append(json_report)
        f.close()

# Create dataframe of annotations. (code_book_label is one of the labels in a report)
df = pd.DataFrame(columns = ['report_id', 'text', 'label', 'start', 'end'])
for rep in reports:
    for label in rep['label']:
        df.loc[len(df)] = [rep['report_id'], rep['data'], label[2], label[0], label[1]] # label[2] is the code_book_label, label[0] is start, label[1] is end

# Remove Non-labels: Addendum, Comment, Question
labels_to_remove = ['Addendum', 'Comment', 'Question']
df = df[~df['label'].isin(labels_to_remove)]

# Remove Leading Metadata
def remove_leading_metadata(text, start, end):
    """
    Removes leading metadata from report text using regex and shifts the start/end indices

    Regex here selects all characters from the start of the document to the end of the line of the last "OBX|"
    This is optionally extended to the first occurrence of a line that contains "Specimen Number", "Tel Number", "Fax Number", or "OrdPhys".
    :param text: report text
    :param start: answer start index
    :param end: answer end index
    :return: report text without metadata, new start, new end
    """
    original_length = len(text)
    new_text = re.sub(
        r"\A.*((\n.*)*OBX\|)?.*((\n.*)*Specimen Number)?.*((\n.*)*Tel Number)?.*((\n.*)*Fax Number)?.*((\n.*)*OrdPhys)?.*",
        "", text)
    removed_length = original_length - len(new_text)
    start = start - removed_length
    end = end - removed_length
    return new_text, start, end

df[['text', 'start', 'end']] = df.apply(lambda row: remove_leading_metadata(row['text'], row['start'], row['end']), axis=1, result_type='expand')

# Load FOI labels (needed to construct questions)
directory = 'U:\Documents\Breast_Non_Synoptic\src'
os.chdir(directory)
foi_lookup_table = pd.read_csv('../results/EDA/FOI_lookup_table.csv',
                         usecols=['label','label_key', 'label_value', 'question'])

# merge df with foi_lookup_table to add questions column to the labels
df = pd.merge(df, foi_lookup_table, on=['label'], how='left')

# Extract Annotated Answer from start/end indices
# extract the span of text that corresponds to "start" and "end" and label as "annotation"
def extract_text(row):
    text = row['text']
    start = row['start']
    end = row['end']
    
    if isinstance(text, str) and isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return text[int(start):int(end)]
    else:
        return np.nan

df['answer'] = df.apply(extract_text, axis=1)

# Split data into train/valid/test by report_id (e.g. report_id 2 will only be in train and not valid or test)
# Get unique report_ids
report_ids = df['report_id'].unique()

# Split report_ids into train, validation, and test sets
train_val_report_ids, test_report_ids = train_test_split(report_ids, test_size=test_split_perc, random_state=42)
train_report_ids, val_report_ids = train_test_split(train_val_report_ids, test_size=val_split_perc/(train_split_perc + val_split_perc), random_state=42)

# Get the rows corresponding to the report_ids in each split
train_data = df[df['report_id'].isin(train_report_ids)]
val_data = df[df['report_id'].isin(val_report_ids)]
test_data = df[df['report_id'].isin(test_report_ids)]

# Subset and split dfs for train data by FOI
unique_keys_train = train_data['label_key'].unique()
print("Creating FOI train data...")
for label_key in unique_keys_train:
    # create subset DataFrame based on key
    train_df_subset = train_data[train_data['label_key'] == label_key]

    # save subset DataFrame to CSV
    train_df_subset.to_csv('../data/clean/non_synoptic/train_FOI/' + 'train_' + label_key + '.csv', index=False)

# Subset and split dfs for val data by FOI
unique_keys = val_data['label_key'].unique()
print("Creating FOI val data...")
for label_key in unique_keys:
    # create subset DataFrame based on key
    val_df_subset = val_data[val_data['label_key'] == label_key]

    # save subset DataFrame to CSV
    val_df_subset.to_csv('../data/clean/non_synoptic/val_FOI/' + 'val_' + label_key + '.csv', index=False)

# save dataframes as csv files
new_directory = 'U:\Documents\Breast_Non_Synoptic\src'
os.chdir(new_directory)

print("Saving data splits...")
train_data.to_csv('../data/clean/non_synoptic/train_data.csv', index=False)
val_data.to_csv('../data/clean/non_synoptic/val_data.csv', index=False)
test_data.to_csv('../data/clean/non_synoptic/test_data.csv', index=False)

# Create combined synoptic train and non_synoptic train data
synoptic_train_df = pd.read_csv('../data/clean/synoptic/synoptic_train.csv')
synoptic_train_df = synoptic_train_df[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
non_synoptic_train_df = train_data[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_train_non_syn_train_df = pd.concat([synoptic_train_df, non_synoptic_train_df], ignore_index= True)
syn_train_non_syn_train_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_train_and_non_synoptic_train.csv', index=False)

# Create combined synoptic val and non_synoptic val data
synoptic_val_df = pd.read_csv('../data/clean/synoptic/synoptic_val.csv')
synoptic_val_df = synoptic_val_df[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
non_synoptic_val_df = val_data[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_val_non_syn_val_df = pd.concat([synoptic_val_df, non_synoptic_val_df], ignore_index= True)
syn_val_non_syn_val_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_val_and_non_synoptic_val.csv', index=False)

################## Create complete dataframes (adding absence cases)

# Get unique report_ids
unique_rid_text = df[['report_id', 'text']].drop_duplicates()

# Get unique questions
foi_ques = df['question'].unique()

# Create a new DataFrame with every combination of report_id, text and question
complete_rid_text_ques = pd.DataFrame([(rid, text, ques) for rid, text in zip(unique_rid_text['report_id'], unique_rid_text['text']) for ques in foi_ques],
                                      columns=['report_id', 'text', 'question'])

# adds the rest of the columns to the injected complete_rid_ques df 
complete_df = pd.merge(complete_rid_text_ques, df, on=['report_id', 'text', 'question'], how='left') # will increase in size because there are duplicated annotated reports (e.g. > 1 for some combination of report_id, text, and question)

# Split data into train/valid/test by report_id (e.g. report_id 2 will only be in train and not valid or test)
# Get unique report_ids
report_ids = complete_df['report_id'].unique()

# Split report_ids into train, validation, and test sets
train_val_report_ids, test_report_ids = train_test_split(report_ids, test_size=test_split_perc, random_state=42)
train_report_ids, val_report_ids = train_test_split(train_val_report_ids, test_size=val_split_perc/(train_split_perc + val_split_perc), random_state=42)

# Get the rows corresponding to the report_ids in each split
train_data = complete_df[complete_df['report_id'].isin(train_report_ids)]
val_data = complete_df[complete_df['report_id'].isin(val_report_ids)]
test_data = complete_df[complete_df['report_id'].isin(test_report_ids)]

#### add label_keys to complete data so we can split it by label_key in train_FOI and val_FOI
directory = 'U:\Documents\Breast_Non_Synoptic\src'
os.chdir(directory)
foi_lookup_table = pd.read_csv('../results/EDA/FOI_lookup_table.csv',
                         usecols=['label','label_key', 'label_value', 'question'])

new_lookup = foi_lookup_table.drop_duplicates(subset = ["label_key", "question"])
train_merged = pd.merge(train_data, new_lookup[['question', 'label_key']], on='question', how='left')
train_data = train_data.reset_index(drop=True)
new_comp_train = train_data.copy()
new_comp_train['label_key'] = train_merged['label_key_y']

val_merged = pd.merge(val_data, new_lookup[['question', 'label_key']], on='question', how='left')
val_data = val_data.reset_index(drop=True)
new_comp_val = val_data.copy()
new_comp_val['label_key'] = val_merged['label_key_y']

#### add No mention label to empty label_value values
# Fill NaN and empty values in the "label_value" column with "No Mention"
# new_comp_train['label_value'].fillna('No Mention', inplace=True)
# new_comp_train['label_value'].replace('', 'No Mention', inplace=True)
# new_comp_val['label_value'].fillna('No Mention', inplace=True)
# new_comp_val['label_value'].replace('', 'No Mention', inplace=True)

# Subset and split dfs for train data by FOI
unique_keys_train = new_comp_train['label_key'].unique()
print("Creating complete FOI train data...")
for label_key in unique_keys_train:
    # create subset DataFrame based on key
    train_df_subset = new_comp_train[new_comp_train['label_key'] == label_key]

    # save subset DataFrame to CSV
    train_df_subset.to_csv('../data/clean/non_synoptic/complete/train_FOI/' + 'train_' + label_key + '.csv', index=False)

# Subset and split dfs for val data by FOI
unique_keys_val = new_comp_val['label_key'].unique()
print("Creating complete FOI val data...")
for label_key in unique_keys_val:
    # create subset DataFrame based on key
    val_df_subset = new_comp_val[new_comp_val['label_key'] == label_key]

    # save subset DataFrame to CSV
    val_df_subset.to_csv('../data/clean/non_synoptic/complete/val_FOI/' + 'val_' + label_key + '.csv', index=False)

# save dataframes as csv files
new_directory = 'U:\Documents\Breast_Non_Synoptic\src'
os.chdir(new_directory)

print("Saving complete data splits...")
new_comp_train.to_csv('../data/clean/non_synoptic/complete/complete_train_data.csv', index=False)
new_comp_val.to_csv('../data/clean/non_synoptic/complete/complete_val_data.csv', index=False)
test_data.to_csv('../data/clean/non_synoptic/complete/complete_test_data.csv', index=False)

# Create combined synoptic train and complete non_synoptic train data
complete_non_synoptic_train_df = new_comp_train[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_train_comp_non_syn_train_df = pd.concat([synoptic_train_df, complete_non_synoptic_train_df], ignore_index= True)
syn_train_comp_non_syn_train_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_train_and_non_synoptic_train_complete.csv')

# Create combined synoptic val and complete non_synoptic val data
complete_non_synoptic_val_df = new_comp_val[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_val_comp_non_syn_val_df = pd.concat([synoptic_val_df, complete_non_synoptic_val_df], ignore_index= True)
syn_val_comp_non_syn_val_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_val_and_non_synoptic_val_complete.csv')
