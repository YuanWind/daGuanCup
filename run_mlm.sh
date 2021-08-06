python code_mlm/convert_official_data.py --train_mode 'unlabel,train,test' --unlabel_data 'data/ori_data/sample_unlabel_data_10000.json'
python code_mlm/process_oov_data.py
python code_mlm/convert_data.py
python code_mlm/train_tokenizer.py
python code_mlm/construct_ngram_dict.py
python code_mlm/train_w2v.py
python code_mlm/construct_ngram_meta_info.py
#python code_mlm/train_mlm.py --batch_size 128 --num_epochs 200