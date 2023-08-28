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

test_split_perc = 0.75 # the percentage of the testing set you want to keep

# load train and test
test_df = pd.read_csv('../data/clean/non_synoptic/test_data.csv')
train_df = pd.read_csv('../data/clean/non_synoptic/train_data.csv')

# Get unique report_ids
test_report_ids = test_df['report_id'].unique()
train_report_ids, new_test_report_ids = train_test_split(test_report_ids, test_size=test_split_perc, random_state=42)

# Get the rows corresponding to the report_ids in each split
new_train_df = test_df[test_df['report_id'].isin(train_report_ids)]
new_test_df = test_df[test_df['report_id'].isin(new_test_report_ids)]

# add new train data to train df
new_train_df = pd.concat([train_df, new_train_df], axis = 0)

# Subset and split dfs for train data by FOI
unique_keys_train = new_train_df['label_key'].unique()
print("Creating FOI train data...")
for label_key in unique_keys_train:
    # create subset DataFrame based on key
    train_df_subset = new_train_df[new_train_df['label_key'] == label_key]

    # save subset DataFrame to CSV
    train_df_subset.to_csv('../data/clean/non_synoptic/train_FOI/' + 'train_' + label_key + '.csv', index=False)

print("Saving new data splits...")
new_train_df.to_csv('../data/clean/non_synoptic/train_data.csv', index=False)
new_test_df.to_csv('../data/clean/non_synoptic/test_data.csv', index=False)

# Create combined synoptic train and non_synoptic train data
print("Creating combined synoptic train and non_synoptic train data...")
synoptic_train_df = pd.read_csv('../data/clean/synoptic/synoptic_train.csv')
synoptic_train_df = synoptic_train_df[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
non_synoptic_train_df = new_train_df[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_train_non_syn_train_df = pd.concat([synoptic_train_df, non_synoptic_train_df], ignore_index= True)
syn_train_non_syn_train_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_train_and_non_synoptic_train.csv', index=False)

# load comp train and test
comp_test_df = pd.read_csv('../data/clean/non_synoptic/complete/complete_test_data.csv')
comp_train_df = pd.read_csv('../data/clean/non_synoptic/complete/complete_train_data.csv')

# Get unique report_ids
comp_test_report_ids = comp_test_df['report_id'].unique()
comp_train_report_ids, new_comp_test_report_ids = train_test_split(comp_test_report_ids, test_size=test_split_perc, random_state=42)

# Get the rows corresponding to the report_ids in each split
new_comp_train_df = comp_test_df[comp_test_df['report_id'].isin(comp_train_report_ids)]
new_comp_test_df = comp_test_df[comp_test_df['report_id'].isin(new_comp_test_report_ids)]

# add new train data to train df
new_comp_train_df = pd.concat([comp_train_df, new_comp_train_df], axis = 0)

#### add label_keys to complete data so we can split it by label_key in train_FOI and val_FOI
directory = 'U:\Documents\Breast_Non_Synoptic\src'
os.chdir(directory)
foi_lookup_table = pd.read_csv('../results/EDA/FOI_lookup_table.csv',
                         usecols=['label','label_key', 'label_value', 'question'])

new_lookup = foi_lookup_table.drop_duplicates(subset = ["label_key", "question"])
train_merged = pd.merge(new_comp_train_df, new_lookup[['question', 'label_key']], on='question', how='left')
train_data = new_comp_train_df.reset_index(drop=True)
new_comp_train = train_data.copy()
new_comp_train['label_key'] = train_merged['label_key_y']

#### add No mention label to empty label_value values
# new_comp_train['label_value'].fillna('No Mention', inplace=True)
# new_comp_train['label_value'].replace('', 'No Mention', inplace=True)

# Subset and split dfs for train data by FOI
unique_keys_train = new_comp_train['label_key'].unique()
print("Creating complete FOI train data...")
for label_key in unique_keys_train:
    # create subset DataFrame based on key
    train_df_subset = new_comp_train[new_comp_train['label_key'] == label_key]

    # save subset DataFrame to CSV
    train_df_subset.to_csv('../data/clean/non_synoptic/complete/train_FOI/' + 'train_' + label_key + '.csv', index=False)


print("Saving new comp data splits...")
new_comp_train.to_csv('../data/clean/non_synoptic/complete/complete_train_data.csv', index=False)
new_comp_test_df.to_csv('../data/clean/non_synoptic/complete/complete_test_data.csv', index=False)

print("Creating complete combined synoptic train and non_synoptic train data...")
# Create combined synoptic train and complete non_synoptic train data
complete_non_synoptic_train_df = new_comp_train[['report_id', 'question', 'text', 'label', 'start', 'end', 'label_key', 'label_value', 'answer']]
syn_train_comp_non_syn_train_df = pd.concat([synoptic_train_df, complete_non_synoptic_train_df], ignore_index= True)
syn_train_comp_non_syn_train_df.to_csv('../data/clean/synoptic_and_non_synoptic/synoptic_train_and_non_synoptic_train_complete.csv')




