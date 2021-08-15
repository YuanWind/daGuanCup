import fire
import sys
sys.path.extend(['../../','../','./'])
from simpletransformers_addons.models.pair_lm.pair_lm_model import LanguageModelingModel, LanguageModelingArgs
import logging
def train_mlm(train_file='data/mlm_data/train_mlm.tsv.str',
              test_file='data/mlm_data/test_mlm.tsv.str',
              model_name='chinese-bert-wwm-ext',
              model_type='bert',
              batch_size=128,
              gradient_accumulation_steps=1,
              num_epochs=200,
              manual_seed=124525601,
              learning_rate=6e-5,
              min_learning_rate=0.0,
              use_relative_segment=False,
              use_fgm=False,
              fgm_epsilon=0.4,
              model_number_id=1,
              out_base_dir='/content'
              ):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = LanguageModelingArgs()
    model_args.manual_seed = manual_seed
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_train_epochs = num_epochs
    model_args.block_size = 256
    model_args.max_seq_length = 256
    model_args.polynomial_decay_schedule_lr_end = min_learning_rate / learning_rate
    model_args.train_batch_size = batch_size
    model_args.gradient_accumulation_steps = gradient_accumulation_steps

    if use_relative_segment:
        model_args.dataset_type = "symmetric_sentence_pair"
    else:
        model_args.dataset_type = "line_by_line"
        # line_by_line
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_steps = 10*867*64/batch_size # 每10轮保存一个模型
    model_args.save_best_model = False
    model_args.evaluate_during_training = False
    model_args.eval_batch_size=16
    model_args.evaluate_during_training_steps = 1000
    model_args.tokenizer_name='tokenizer'
    model_args.scheduler='cosine_schedule_with_warmup'
    model_args.warmup_steps=867*64
    import os
    if use_relative_segment:
        test_dir = os.path.join('mlm', model_name + '_symmetric')
    else:
        if use_fgm:
            test_dir = os.path.join('mlm', model_name + '_fgm')
        else:
            if model_number_id <= 1:
                test_dir = os.path.join('mlm', model_name)
            else:
                test_dir = os.path.join('mlm', model_name + f'{model_number_id}')
    model_args.tensorboard_dir = os.path.join('logs', 'mlm_{}'.format(model_name))
    model_args.cache_dir = os.path.join(out_base_dir,test_dir, 'cached')
    model_args.output_dir = os.path.join(out_base_dir,test_dir, 'outputs')
    model_args.best_model_dir = os.path.join(test_dir, 'best_model')

    model_args.learning_rate = learning_rate

    if use_relative_segment:
        model_args.config = {
            "segment_type": "relative"
        }

    model = LanguageModelingModel(
        model_type, os.path.join('hfl', model_name), args=model_args
    )
    if use_fgm:
        from simpletransformers_addons.model_wrappers import FGMWrapper
        model = FGMWrapper(model, epsilon=fgm_epsilon)

    # Train the model
    model.train_model(train_file, eval_file=test_file)


if __name__ == '__main__':
    fire.Fire(train_mlm)
