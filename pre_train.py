import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)
import tokenizers
def get_vocab():
    # 创建分词器
    bwpt = tokenizers.BertWordPieceTokenizer()
    filepath = "data/processed/train_test.txt" # 语料文件
    #训练分词器
    bwpt.train(
        files=[filepath],
        vocab_size=50000, # 这里预设定的词语大小不是很重要
        min_frequency=1,
        limit_alphabet=1000
    )
    # 保存训练后的模型词表
    bwpt.save_model('./pretrained_models/')
    # 加载刚刚训练的tokenizer
# tokenizer=BertTokenizer(vocab_file='./pretrained_models/vocab.txt')
# print(tokenizer('254 121'))
def get_model():
    token_path='./pretrained_models/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(token_path, do_lower_case=True)
    # 自己修改部分配置参数
    config_kwargs = {
        "revision": 'main',
        "use_auth_token": None,
        #      "hidden_size": 512,
        #     "num_attention_heads": 4,
        "hidden_dropout_prob": 0.2,
        #     "vocab_size": 863 # 自己设置词汇大小
    }
    # 将模型的配置参数载入
    base_model='bert-base-uncased'
    config = AutoConfig.from_pretrained(base_model, **config_kwargs)
    # 载入预训练模型
    model = AutoModelForMaskedLM.from_pretrained(
        base_model,
        config=config,
        cache_dir=None,
        revision='main',
        use_auth_token=None,
    )
    model.resize_token_embeddings(len(tokenizer))
    # 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='data/processed/train_test.txt', block_size=128)
    # MLM模型的数据DataCollator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # 训练参数
    pretrain_batch_size = 4
    num_train_epochs = 5
    training_args = TrainingArguments(output_dir='./tmp/',num_train_epochs=num_train_epochs, learning_rate=6e-5,per_device_train_batch_size=pretrain_batch_size, save_total_limit = 10,save_steps=10000)  #
    # 通过Trainer接口训练模型
    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

    # 开始训练
    trainer.train()
    trainer.save_model('./pretrained_models/')
# get_vocab()
get_model()