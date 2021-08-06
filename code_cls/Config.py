from transformers import TrainingArguments
import logging


DEAUFALT_CONFIG = {
    'pre_model_file': 'mlm/chinese-bert-wwm-ext/outputs/',
    'train_file': 'data/processed/train.pkl',
    'dev_file': 'data/processed/dev.pkl',
    'test_file': 'data/processed/test.pkl',
    'vocab_file': 'saved/vocab.pkl',
    'padding': 'max_length',
    'max_length':256,
    'save_dir': 'saved',
    'best_model_dir': 'saved_models',
    'output_dir': 'tmp/',
    'num_train_epochs': 2,
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'do_train': True,
    'do_eval': True,
    'evaluation_strategy': 'steps',
    'eval_steps': 200,
    'save_total_limit': 5,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'log_file': ''
}


class Config:
    def __init__(self, **extra_args):
        self.extra_args = extra_args
        for k, v in extra_args.items():
            DEAUFALT_CONFIG[k] = v


        self.pre_model_file = DEAUFALT_CONFIG.get('pre_model_file')
        self.train_file = DEAUFALT_CONFIG.get('train_file')
        self.dev_file = DEAUFALT_CONFIG.get('dev_file')
        self.test_file = DEAUFALT_CONFIG.get('test_file')
        self.vocab_file = DEAUFALT_CONFIG.get('vocab_file')
        self.padding = DEAUFALT_CONFIG.get('padding')
        self.max_length=DEAUFALT_CONFIG.get('max_length')
        self.save_dir = DEAUFALT_CONFIG.get('save_dir')
        self.best_model_dir = DEAUFALT_CONFIG.get('best_model_dir')
        self.log_file=DEAUFALT_CONFIG.get('log_file')


    def train_args(self):
        logging.info('Config parms value list:')
        for k, v in DEAUFALT_CONFIG.items():
            logging.info('{}={}'.format(k, v))
        res = TrainingArguments(
            output_dir=DEAUFALT_CONFIG.get('output_dir'),
            num_train_epochs=DEAUFALT_CONFIG.get('num_train_epochs'),
            learning_rate=DEAUFALT_CONFIG.get('learning_rate'),
            per_device_train_batch_size=DEAUFALT_CONFIG.get('per_device_train_batch_size'),
            per_device_eval_batch_size=DEAUFALT_CONFIG.get('per_device_eval_batch_size'),
            do_train=DEAUFALT_CONFIG.get('do_train'),
            do_eval=DEAUFALT_CONFIG.get('do_eval'),
            evaluation_strategy=DEAUFALT_CONFIG.get('evaluation_strategy'),
            eval_steps=DEAUFALT_CONFIG.get('eval_steps'),
            save_total_limit=DEAUFALT_CONFIG.get('save_total_limit'),
            load_best_model_at_end=DEAUFALT_CONFIG.get('load_best_model_at_end'),
            metric_for_best_model=DEAUFALT_CONFIG.get('metric_for_best_model')

        )
        return res
