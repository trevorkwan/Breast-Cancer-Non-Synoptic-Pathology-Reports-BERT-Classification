@echo off

set BASE_MODEL=bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
set OUTPUT_DIR=U:\Documents\Breast_Non_Synoptic\results\pretrained\bluebert_pubmed_mimic_uncased_squadv2\

python run_qa.py ^
  --model_name_or_path  %BASE_MODEL% ^
  --dataset_name squad_v2 ^
  --do_train ^
  --do_eval ^
  --version_2_with_negative ^
  --per_device_train_batch_size 16 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 3 ^
  --max_seq_length 480 ^
  --doc_stride 64 ^
  --weight_decay 0.01 ^
  --output_dir %OUTPUT_DIR%
