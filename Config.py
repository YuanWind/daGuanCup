from transformers import logging, TrainingArguments,HfArgumentParser

from utils.utils import load_json
logger = logging.get_logger(__name__)


class Config:
    def __init__(self,**extra_args):
        self.extra_args=extra_args

        self.pre_model_file=extra_args.get('pre_model_file','pretrained_models/')
        self.train_file=extra_args.get('pre_model_file','data/processed/train.pkl')
        self.dev_file=extra_args.get('pre_model_file','data/processed/dev.pkl')
        self.test_file=extra_args.get('pre_model_file','data/processed/test.pkl')
        self.vocab_file=extra_args.get('pre_model_file','saved/vocab.pkl')
        self.padding=extra_args.get('pre_model_file','longest')
        self.save_dir=extra_args.get('pre_model_file','saved/')
        self.best_model_dir=extra_args.get('pre_model_file','saved_models/')

    def train_args(self):

        res=TrainingArguments(
            output_dir=self.extra_args.get('output_dir','./tmp/'),
            num_train_epochs= self.extra_args.get('num_train_epochs',2),
            learning_rate=self.extra_args.get('learning_rate',2e-5),
            per_device_train_batch_size=self.extra_args.get('per_device_train_batch_size',2),
            per_device_eval_batch_size=self.extra_args.get('per_device_eval_batch_size',4),
            do_train=self.extra_args.get('do_train',True),
            do_eval=self.extra_args.get('do_eval',True),
            evaluation_strategy=self.extra_args.get('evaluation_strategy','steps'),
            eval_steps=self.extra_args.get('eval_steps',200),
            save_total_limit=self.extra_args.get('save_total_limit',5),
            load_best_model_at_end=self.extra_args.get('load_best_model_at_end',True),
            metric_for_best_model=self.extra_args.get('metric_for_best_model','f1')

        )
        return res