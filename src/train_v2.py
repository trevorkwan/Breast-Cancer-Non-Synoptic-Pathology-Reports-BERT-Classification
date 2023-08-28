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

# Set the seed
set_seed(42)

# Define the arguments
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 5 # default is 3
weight_decay = 0.01
max_seq_length = 480 # The maximum length of a feature (question and context)
doc_stride =  64 # The authorized overlap between two part of the context when splitting it is needed.
question_column_name = "question"
context_column_name = "text"
answer_column_name = "answer"
start_column_name = "start"
end_column_name = "end"
max_train_samples = None
max_eval_samples = None

model_checkpoint = 'franklu/pubmed_bert_squadv2'
model_dir = "../results/trained\\"
model_version = '_v4'
train_split_perc = "40%"
val_split_perc = "30%"
test_split_perc = "40%"
downsample_percentage = "0%, No downsampling."
comment = 'changed train epochs to 5 (v4). frank model fine-tuned on complete non-synoptic data + synoptic train data for all 40 FOI. this has the remove metadata added. this is trained on both non-synoptic and synoptic data. we use 40 percent of non-synoptic data for train.'

# Load the Data
train_file_path = "../data/clean/synoptic_and_non_synoptic/synoptic_train_and_non_synoptic_train_complete.csv"
val_file_path = "../data/clean/synoptic_and_non_synoptic/synoptic_val_and_non_synoptic_val_complete.csv"
train_df = pd.read_csv(train_file_path)
val_df = pd.read_csv(val_file_path)

if max_train_samples is not None:
    # subset for testing purposes
    max_train_samples = min(len(train_df), max_train_samples)
    train_df = train_df.iloc[:max_train_samples]

print("Train_df:", train_df)
if train_df[answer_column_name].isnull().any():
    print("The train_df answer column contains NaN values.")

if max_eval_samples is not None:
    # subset for testing purposes
    max_eval_samples = min(len(val_df), max_eval_samples)
    val_df = val_df.iloc[:max_eval_samples]

print("Val_df:", val_df)
if val_df[answer_column_name].isnull().any():
    print("The val_df answer column contains NaN values.")

train_nrows = len(train_df)
print("Train sample size:", train_nrows)
val_nrows = len(val_df)
print("Val sample size:", val_nrows)

# Define model signature
model_signature = '_' + str(train_nrows) + model_version

# Define the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint) # Is there a difference between these ?
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]

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

pad_on_right = tokenizer.padding_side == "right" # Some models expect padding on the left. Make sure they we account for that

def preprocess_text(df):

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.

    tokenized_data = tokenizer(
        list(df[question_column_name if pad_on_right else context_column_name]),
        list(df[context_column_name if pad_on_right else question_column_name]),
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_data.pop("offset_mapping")

    # Let's label those examples!
    tokenized_data["start_positions"] = []
    tokenized_data["end_positions"] = []

    for i, offsets in enumerate(offset_mapping): # loop through each subspan of text
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_data["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_data.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = df[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if answer is None:
            tokenized_data["start_positions"].append(cls_index)
            tokenized_data["end_positions"].append(cls_index)

        else:
            # Start/end character index of the answer in the text.
            start_char = df[start_column_name][sample_index]
            end_char = df[end_column_name][sample_index]
            # end_char = start_char + len(df[answer_column_name][sample_index])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # The current sub-span being considered corresponds to character indices: offsets[token_start_index][0] to offsets[token_end_index][1] in the original text
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_data["start_positions"].append(cls_index)
                tokenized_data["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_data["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_data["end_positions"].append(token_end_index + 1)

    return tokenized_data

# Fill NaN context values with 
train_processed = preprocess_text(train_df)
validation_processed = preprocess_text(val_df)


train_dataset = Dataset.from_dict({'input_ids': train_processed['input_ids'],
                                    'attention_mask': train_processed['attention_mask'],
                                    'start_positions': train_processed['start_positions'],
                                    'end_positions': train_processed['end_positions']})

validation_dataset = Dataset.from_dict({'input_ids': validation_processed['input_ids'],
                                    'attention_mask': validation_processed['attention_mask'],
                                    'start_positions': validation_processed['start_positions'],
                                    'end_positions': validation_processed['end_positions']})

data_collator = default_data_collator

# change diretory to results/trained// folder
# os.chdir(model_dir)

##### Train Model #####
training_args = TrainingArguments(
    output_dir = model_dir + model_name + model_signature, 
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
    training_args,
    train_dataset= train_dataset,
    eval_dataset= validation_dataset,
    data_collator=data_collator,
    tokenizer= tokenizer,
)
import time
start = time.time()
trainer.train()
elapsed = time.time() - start

# save the model
trainer.save_model(model_dir + model_name + model_signature)

# Define the model card content as a string
model_card_content = f'''Base Model: {model_checkpoint}
Model Name: {model_name}
Model Version (Signature): {model_signature}
Task: QA
Number of Train Samples: {train_nrows}
Numnber of Val Samples: {val_nrows}
Downsample Percentage: {downsample_percentage}
Learning Rate: {learning_rate}
Number of Train Epochs: {num_train_epochs}
Weight Decay: {weight_decay}
Max Seq Length: {max_seq_length}
Doc Stride: {doc_stride}
Train Batch Size: {per_device_train_batch_size}
Validation Batch Size: {per_device_eval_batch_size}
Train Split Percentage: {train_split_perc}
Validation Split Percentage: {val_split_perc}
Test Split Percentage: {test_split_perc}
Train File Path:{train_file_path}
Val File Path: {val_file_path}
Comment: {comment}
'''

# Define the file path for the model card text file
model_card_file_path = model_dir + model_name + model_signature + r'\model_card.txt'

# Write the model card content to the text file
with open(model_card_file_path, 'w', encoding='utf-8') as file:
    file.write(model_card_content)

print(f"Model card has been saved to: {model_card_file_path}")