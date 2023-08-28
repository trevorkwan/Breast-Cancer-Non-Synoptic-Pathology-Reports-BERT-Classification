# import libraries

import os
import re
import json
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import helper functions
from helper_functions import fix_margin, trim_whitespace_from_annotation, fix_histologic_type, update_annotation_indices

# load train data 
df = pd.read_csv('../data/clean/non_synoptic/train_data.csv')

##### Get Counts of Labels 
label_counts = df['label'].value_counts()
# Convert the label counts to a DataFrame
label_counts_df = label_counts.reset_index()
label_counts_df.columns = ['label', 'count']
# Save the label counts DataFrame to a CSV file
label_counts_df.to_csv('../results/EDA/label_counts.csv', index=False)

##### Get Counts of Label Keys and Plot
label_key_counts = df['label_key'].value_counts()
# Convert the label key counts to a DataFrame
label_key_counts_df = label_key_counts.reset_index()
label_key_counts_df.columns = ['label_key', 'count']
# Save the label key counts DataFrame to a CSV file
label_key_counts_df.to_csv('../results/EDA/label_key_counts.csv', index=False)

# Plotting the label key counts in a histogram
label_key_counts_sorted = label_key_counts.sort_values()
plt.figure(figsize=(10,8))
plt.barh(label_key_counts_sorted.index, label_key_counts_sorted.values)
plt.xlabel('Counts')
plt.ylabel('FOI')
plt.title('Field of Interest (FOI) Counts')
plt.yticks(rotation = 0, fontsize = 5)
directory = '../results/EDA/img'  # replace with your desired directory
filename = 'label_key_counts_histogram.png'
full_path = f'{directory}/{filename}'
plt.savefig(full_path, format = 'png')

plt.show()

##### Get Counts of Label Keys per Report and Plot
# Count the number of unique report_ids each label appears in
label_key_report_counts_df = df.groupby('label_key')['report_id'].nunique().reset_index(name='report_counts').sort_values(by = "report_counts", ascending = False)
# rename the columns
label_key_report_counts_df.columns = ['label_key', 'report_counts']
# Save the label report counts DataFrame to a CSV file
label_key_report_counts_df.to_csv('../results/EDA/label_report_counts.csv', index=False)

# plot the label key REPORT counts in a histogram
label_key_report_counts_sorted = label_key_report_counts_df.sort_values(by = "report_counts", ascending = True)
plt.figure(figsize=(10,8))
plt.barh(label_key_report_counts_sorted.label_key, label_key_report_counts_sorted.report_counts)
plt.xlabel('Report Counts')
plt.ylabel('FOI')
plt.title('Field of Interest (FOI) Report Counts')
plt.yticks(rotation = 0, fontsize = 5)
directory = '../results/EDA/img'  # replace with your desired directory
filename = 'label_key_report_counts_histogram.png'
full_path = f'{directory}/{filename}'
plt.savefig(full_path, format = 'png')

plt.show()
