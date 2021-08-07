export TOKENIZERS_PARALLELISM=true
nohup python -u  code_cls/main.py train \
								  --pre_model_file 'mlm/chinese-bert-wwm-ext/outputs/checkpoint-160000' \
								  --num_train_epochs 20 \
								  --train_batch_size 16 \
								  --eval_batch_size 64 \
> 'tmp/cls.log' 2>&1 &