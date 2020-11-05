DATA_DIR=/path/to/data_dir
# please set your data_dir, like ~/data/convbert
NAME=convbert_medium-small

python3 run_finetuning.py --data-dir $DATA_DIR \
--model-name $NAME --hparams '{"model_size": "medium-small", "task_names": ["cola"]}'