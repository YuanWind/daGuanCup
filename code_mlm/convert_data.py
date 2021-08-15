import json
import logging
import os
import fire
import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig()
logger = logging.getLogger('convert_data')
logger.setLevel(logging.INFO)

def construct_vocab(output_vocab_file):
    # construct vocab
    vocab = {
        'num2token': {},
        'token2num': {}
    }
    num_id = 0
    for token in list(range(0x4e00, 0x9fa6)) + list(range(0x0800, 0x4e00)):
        if chr(token) not in vocab['token2num']:
            vocab['num2token'][num_id] = chr(token)
            vocab['token2num'][chr(token)] = num_id
            num_id += 1
    with open(output_vocab_file, 'w', encoding='utf-8') as fout:
        json.dump(vocab, fout, ensure_ascii=False, indent=2)
    return vocab

def convert_record_style(input_file, vocab, output_file):
    input_df = pd.read_csv(input_file, header=None, delimiter='\t')
    def convert_str_style(input_str):
        token_ids = [int(item) for item in input_str.split() if item]
        tokens = [vocab['num2token'][idx] for idx in token_ids]
        return ''.join(tokens)
    input_df[0] = input_df[0].apply(convert_str_style)
    input_df.to_csv(output_file, header=None, sep='\t', index=False)

def construct_tokenizer_data(input_file, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        with open(input_file, encoding='utf-8') as fin:
            for line in tqdm(fin):
                fout.write(line)


def convert_data(train_file='data/mlm_data/train_mlm.tsv',
                 test_file='data/mlm_data/test_mlm.tsv',
                 output_dir='data/mlm_data',
                 ):
    os.makedirs(output_dir, exist_ok=True)
    logger.info('construct vocabulary...')
    vocab = construct_vocab(output_vocab_file=os.path.join(output_dir, 'vocab.json'))

    logger.info('convert ids record to string')
    convert_record_style(train_file, vocab, train_file + '.str')
    convert_record_style(test_file, vocab, test_file + '.str')

    logger.info('constuct tokenizer training data')
    construct_tokenizer_data(os.path.join(train_file + '.str'),os.path.join(output_dir, 'tokenizer_data.txt'))


if __name__ == '__main__':
    fire.Fire(convert_data)
