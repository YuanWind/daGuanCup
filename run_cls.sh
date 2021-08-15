export TOKENIZERS_PARALLELISM=true
nohup python code_cls/main.py train --output_dir 'tmp_cls/' \
                 --num_train_epochs 20 \
                 --train_batch_size 16 \
                 --eval_batch_size 64 \
                 --pre_model_file '/content/mlm/chinese-bert-wwm-ext/outputs' \
> 'tmp_cls/cls.log' 2>&1 &