DATA_DIR=/path/to/data_dir
# please set your data_dir, like ~/data/convbert
NAME=convbert_medium-small
python3 run_pretraining.py --data-dir $DATA_DIR --model-name $NAME --hparams '{"model_size": "medium-small"}'
# The small-sized and medium-small-sized ConvBERT model can run on a V100 GPU, while the based-sized model needs more computation resources.