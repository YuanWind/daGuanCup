python mlm_train/convert_official_data.py --train_mode 'unlabel,train,test'
python mlm_train/process_oov_data.py
python mlm_train/convert_data.py
python mlm_train/train_tokenizer.py
python mlm_train/construct_ngram_dict.py
python mlm_train/train_w2v.py
python mlm_train/construct_ngram_meta_info.py
#python mlm_train/train_mlm.py --batch_size 128 --num_epochs 200