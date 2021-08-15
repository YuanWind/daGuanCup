from transformers import TrainingArguments
import logging
logging.basicConfig()
logger = logging.getLogger('config')
logger.setLevel(logging.INFO)

DEAUFALT_CONFIG = {
    'pre_model_file': 'model_mlm',
    'train_file': 'data/processed/train.pkl',
    'dev_file': 'data/processed/dev.pkl',
    'test_file': 'data/processed/test.pkl',
    'vocab_file': 'saved_files/my_vocab.pkl',
    'padding': 'longest',
    'max_length':256,
    'save_dir': 'saved_files',
    'best_model_dir': 'model_cls',
    'output_dir': '/content/tmp_cls',
    'num_train_epochs': 2,
    'learning_rate': 2e-5,
    'train_batch_size': 2,
    'eval_batch_size': 2,
    'do_train': True,
    'do_eval': True,
    'evaluation_strategy': 'steps',
    'eval_steps': 200,
    'save_steps': 200,
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'log_dir': 'logs/cls'
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
        self.log_dir=DEAUFALT_CONFIG.get('log_dir')


    def train_args(self):
        logger.info('Config parms value list:')
        for k, v in DEAUFALT_CONFIG.items():
            logger.info('{}={}'.format(k, v))
        res = TrainingArguments(
            output_dir=DEAUFALT_CONFIG.get('output_dir'),
            num_train_epochs=DEAUFALT_CONFIG.get('num_train_epochs'),
            learning_rate=DEAUFALT_CONFIG.get('learning_rate'),
            per_device_train_batch_size=DEAUFALT_CONFIG.get('train_batch_size'),
            per_device_eval_batch_size=DEAUFALT_CONFIG.get('eval_batch_size'),
            do_train=DEAUFALT_CONFIG.get('do_train'),
            do_eval=DEAUFALT_CONFIG.get('do_eval'),
            evaluation_strategy=DEAUFALT_CONFIG.get('evaluation_strategy'),
            eval_steps=DEAUFALT_CONFIG.get('eval_steps'),
            save_steps=DEAUFALT_CONFIG.get('save_steps'),
            save_total_limit=DEAUFALT_CONFIG.get('save_total_limit'),
            load_best_model_at_end=DEAUFALT_CONFIG.get('load_best_model_at_end'),
            metric_for_best_model=DEAUFALT_CONFIG.get('metric_for_best_model'),
            report_to='tensorboard',
            logging_dir=DEAUFALT_CONFIG.get('log_dir')

        )
        return res
