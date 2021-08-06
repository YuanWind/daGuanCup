# 只用官方的train.tsv和test.tsv训练语言模型
python mlm_train/convert_official_data.py --train_mode 'train,test'
python mlm_train/process_oov_data.py
python mlm_train/convert_data.py
python mlm_train/train_tokenizer.py
python mlm_train/construct_ngram_dict.py
python mlm_train/train_w2v.py
python mlm_train/construct_ngram_meta_info.py
python mlm_train/train_mlm.py