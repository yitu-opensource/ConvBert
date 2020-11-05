DATA_DIR=/path/to/data_dir
# please set your data_dir, like ~/data/convbert

# extract data
tar xf openwebtext.tar.xz
# move to data_dir
mv openwebtext $DATA_DIR/openwebtext
cp vocab.txt $DATA_DIR/vocab.txt
# build pre-train tf-record 
python3 build_openwebtext_pretraining_dataset.py --data-dir $DATA_DIR --num-processes 5
