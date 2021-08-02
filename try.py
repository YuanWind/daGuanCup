import fire
import numpy as np
import pandas
from sklearn.metrics import precision_recall_fscore_support
from transformers import *
from Config import Config
from utils.Dataloader import MyDataSet, Vocab
from utils.utils import load_pkl, write_pkl
import os
logger = logging.get_logger(__name__)

def convert_example_to_features(inst, tokenizer, vocab):
    features = {'input_ids': None, 'attention_mask': None, 'token_type_ids': None, 'label_ids': None}
    sentence = ' '.join(inst.sentence)
    label_12 = inst.label_12

    token_out = tokenizer(sentence)
    features['input_ids'] = token_out['input_ids']
    features['attention_mask'] = token_out['attention_mask']
    features['token_type_ids'] = token_out['token_type_ids']

    if label_12 is not None:
        label_ids = vocab.label2id[label_12]
        features['label_ids'] = label_ids
    else:
        label_ids = vocab.label2id['PAD']
        features['label_ids'] = label_ids
    return features


def compute_metrics(p: EvalPrediction):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), preds.flatten(), average='macro', zero_division=0)
    return {
        'accuracy': (preds == p.label_ids).mean(),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def set_args(**args):
    config = Config(**args)
    training_args = config.train_args()
    return config, training_args


def get_data(cfg):
    model_path = cfg.pre_model_file
    train_data = load_pkl(cfg.train_file)  # list类型
    dev_data = load_pkl(cfg.dev_file)  # list 类型
    if not os.path.isfile(cfg.vocab_file):
        vocab = Vocab()
        vocab.build(train_data)
        write_pkl(vocab, cfg.vocab_file)
    else:
        vocab = load_pkl(cfg.vocab_file)
    # token_path='./pretrained_models/vocab.txt'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_samples = [convert_example_to_features(inst, tokenizer, vocab) for inst in train_data]
    train_dataset = MyDataSet(train_samples)

    dev_samples = [convert_example_to_features(inst, tokenizer, vocab) for inst in dev_data]
    dev_dataset = MyDataSet(dev_samples)

    test_data = load_pkl(cfg.test_file)
    test_samples = [convert_example_to_features(inst, tokenizer, vocab) for inst in test_data]
    test_dataset = MyDataSet(test_samples)
    return vocab, tokenizer, train_dataset, dev_dataset, test_dataset


def build_model(model_path, num_labels):
    model_cfg = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_cfg)
    return model


def train(**args):
    cfg, training_args = set_args(**args)
    vocab, tokenizer, train_dataset, dev_dataset, test_dataset = get_data(cfg)
    num_labels = len(vocab.label2id)
    model = build_model(cfg.pre_model_file, num_labels)

    data_collator = DataCollatorWithPadding(tokenizer, padding=cfg.padding)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(cfg.save_dir)

    dev_res = trainer.evaluate(dev_dataset)
    print(dev_res)
    predict_res = trainer.predict(test_dataset)
    write_pkl(predict_res, cfg.save_dir + '/predict_res.pkl')
    preds = predict_res[0]
    pred_labels = np.argmax(preds, axis=-1)
    res = {'id': [], 'label': []}
    test_data = load_pkl(cfg.test_file)
    for index, inst in enumerate(test_data):
        idx = inst.idx,
        label = pred_labels[index]
        res['id'].append(idx)
        res['label'].append(label)
    res = pandas.DataFrame(res)
    res.to_csv('res.csv', index=False)

def predict(**args):
    cfg, training_args = set_args(**args)
    test_data = load_pkl(cfg.test_file)
    vocab, tokenizer, train_dataset, dev_dataset, test_dataset = get_data(cfg)
    num_labels = len(vocab.label2id)
    model = build_model(cfg.best_model_dir, num_labels)
    data_collator = DataCollatorWithPadding(tokenizer, padding=cfg.padding)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=dev_dataset, compute_metrics=compute_metrics)
    # 评测模型并生成测试集提交文件
    dev_res = trainer.evaluate(dev_dataset)
    print(dev_res)
    predict_res = trainer.predict(test_dataset)
    write_pkl(predict_res, cfg.save_dir + '/predict_res.pkl')
    preds = predict_res[0]
    pred_labels = np.argmax(preds, axis=-1)
    res = {'id': [], 'label': []}
    for index, inst in enumerate(test_data):
        idx = inst.idx,
        label = pred_labels[index]
        res['id'].append(idx)
        res['label'].append(label)
    res = pandas.DataFrame(res)
    res.to_csv('res.csv', index=False)


if __name__ == '__main__':
    train(num_train_epochs=20)
    # fire.Fire(train)
    # fire.Fire({'train': train, 'pred': predict})
