export TOKENIZERS_PARALLELISM=true
python code_mlm/convert_official_data.py --train_mode 'unlabel,train,test' --unlabel_data 'data/ori_data/sample_unlabel_data_10000.json'
python code_mlm/process_oov_data.py
python code_mlm/convert_data.py
python code_mlm/train_tokenizer.py
python code_mlm/construct_ngram_dict.py --min_frequence 10
python code_mlm/train_w2v.py --epochs 100
python code_mlm/construct_ngram_meta_info.py
nohup python code_mlm/train_mlm.py --batch_size 50 --num_epochs 50 --out_base_dir 'tmp_mlm' > 'tmp_mlm/mlm.log' 2>&1 &