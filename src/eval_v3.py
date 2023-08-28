import pandas as pd
import numpy as np
import re
import os

from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer, set_seed

# Set the seed
set_seed(42)

# Define the arguments
model_checkpoint = 'franklu/pubmed_bert_squadv2'
model_name = model_checkpoint.split("/")[-1]
model_dir = "../results/trained\\"
model_signature = '_19827_v3\\'
version = 'v2'

score_threshold = '0.01'
comment = f'made predictions based on validation data, each validation FOI includes absent cases, 30% validation data. if <{score_threshold} QA model confidence score, return ""'

question_column_name = "question"
context_column_name = "text"
answer_column_name = "answer"

# List of FOIs that you want to evaluate
desired_fois = ["DCIS Margins", "ER Status", "Extranodal Extension", "HER2 Status", "Insitu Component", "Invasive Carcinoma", "Invasive Carcinoma Margins", "Lymphovascular Invasion", "Necrosis", "PR Status", "Tumour Focality"]

# Load model & Trainer from saved checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_dir + model_name + model_signature)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
qa_pipe = pipeline('question-answering', model= model, tokenizer= tokenizer)

# Because we need to first categorize the parent FOI's, we need to process one report at a time
def model_inference(qa_pipe, df):
    total_rows = df.shape[0]
    print(f"Total rows in dataframe: {total_rows}")
    for idx, row in df.iterrows():
        print(f"Processing row {idx+1} of {total_rows}")
        query = qa_pipe(row[question_column_name], row[context_column_name]) # e.g. qa_pipe('What is the invasive carcinoma?', 'MSH|^~\\&|FHA-MT|FHA-MT^SCSC^LABLOCATION|eMaRC|...)
        query_df = pd.DataFrame(query, index = [0]).query(f'score > {score_threshold}') # Use the score_threshold variable here

        # If no QA model predictions have score greater than the minimum, return ""
        if len(query_df) == 0:
            query_df2 = pd.DataFrame(query, index=[0])
            max_score_index = query_df2['score'].idxmax()
            df.at[idx, 'qa_score'] = query_df2['score'][max_score_index]
            df.at[idx, 'qa_answer'] = ""

        # If only a single extraction, then just update the df with score and answer
        elif len(query_df) == 1:
            df.at[idx, 'qa_score'] = query_df['score'][0]
            df.at[idx, 'qa_answer'] = query_df['answer'][0]

    return(df)

def process_file(file_path):
    df = pd.read_csv(file_path)
    foi = os.path.splitext(os.path.basename(file_path))[0].replace('val_', '')  # get FOI from file name
    if foi not in desired_fois:  # skip this file if not in the list of desired FOIs
        return
    print(f"Processing file: {file_path}, FOI: {foi}")
    
    # Run the model inference 
    predictions = model_inference(qa_pipe, df)
    
    # Save the DataFrame to a CSV file
    output_dir = os.path.join(model_dir, model_name + model_signature, "eval", version, foi)

    # Create the output directory (foi folder) if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions.to_csv(os.path.join(output_dir,'predictions_' + foi + '.csv'), index=False)

# Get all csv files in directory
dir_path = '../data/clean/non_synoptic/complete/val_FOI/'
files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]

for file in files:
    process_file(file)

# Define the eval card content as a string
eval_card_content = f'''Max Confidence Score Threshold: {score_threshold}
Comment: {comment}
'''

# Define the file path for the model card text file
output_dir = os.path.join(model_dir, model_name + model_signature, "eval", version)
eval_card_file_path = output_dir + r'\eval_card.txt'

# Write the model card content to the text file
with open(eval_card_file_path, 'w', encoding='utf-8') as file:
    file.write(eval_card_content)

print(f"Eval card has been saved to: {eval_card_file_path}")
