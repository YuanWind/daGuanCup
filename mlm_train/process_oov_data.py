import fire
import json
import logging
import os
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

logging.basicConfig()
logger = logging.getLogger('convert_data')
logger.setLevel(logging.INFO)

def process_oov_record(record, normal_vocab,
                       idmap,
                       min_frequence=5,
                       min_oov_word_idx=35000):
    text_a = record
    tokens_a = text_a.split()
    oov_word_map = {}
    cur_oov_idx = min_oov_word_idx
    for tokens in [tokens_a]:
        for i in range(len(tokens)):
            if normal_vocab.get(tokens[i], 0) < min_frequence:
                if tokens[i] not in oov_word_map:
                    oov_word_map[tokens[i]] = str(cur_oov_idx)
                    cur_oov_idx += 1
                tokens[i] = oov_word_map[tokens[i]]
            else:
                tokens[i] = idmap[tokens[i]]
    return " ".join(tokens_a)


def construct_normal_vocab(input_file,
                           output_file,
                           output_idmap_file):
    input_df = pd.read_csv(input_file, header=None, delimiter='\t')
    word_ct = Counter()
    normal_voc = {}
    for _, row in tqdm(input_df.iterrows()):
        text_a = row[0]
        word_ct.update(text_a.split())
    idmap = {}
    for idx, (word, num) in enumerate(word_ct.most_common()):
        normal_voc[word] = num
        idmap[word] = str(idx)
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(normal_voc, fout)
    with open(output_idmap_file, 'w', encoding='utf-8') as fout:
        json.dump(idmap, fout, ensure_ascii=False, indent=2)
    return normal_voc, idmap

def process_oov_df(input_df, normal_vocab, idmap, min_frequence=3):
    input_df = input_df.copy(deep=True)
    for idx, row in tqdm(input_df.iterrows()):
        text_a = row[0]
        text_a = process_oov_record(text_a, normal_vocab,
                                            idmap,
                                            min_frequence=min_frequence)
        input_df.at[idx, 0] = text_a
    return input_df

def process_oov_file(input_file,
                     output_file,
                     normal_vocab,
                     idmap,
                     mode='train'):
    input_df = pd.read_csv(input_file, header=None, delimiter='\t')
    # if mode == 'train':
    #     output_df1 = process_oov_df(input_df, normal_vocab, min_frequence=3)
    #     output_df2 = process_oov_df(input_df, normal_vocab, min_frequence=1)
    #     output_df = pd.concat([output_df1, output_df2])
    #     output_df = output_df.drop_duplicates()
    # else:
    #     output_df = process_oov_df(input_df, normal_vocab, min_frequence=1)
    output_df = process_oov_df(input_df, normal_vocab, idmap, min_frequence=3)
    output_df.to_csv(output_file, header=None, sep='\t', index=False)

def process_oov_words(unlabel_file='data/mlm_data/unlabel_data.tsv',
                      train_file='data/mlm_data/train.tsv',
                      test_file='data/mlm_data/test.tsv',
                      output_dir='data/mlm_data'):
    os.makedirs(output_dir, exist_ok=True)
    logger.info('construct normal vocabulary...')
    normal_vocab, idmap = construct_normal_vocab(train_file,
                                          os.path.join(output_dir, 'normal_vocab.json'),
                                          os.path.join(output_dir, 'idmap.json'))
    logger.info('process oov file...')
    process_oov_file(unlabel_file, os.path.join(output_dir, 'unlable_mlm.tsv'), normal_vocab, idmap,
                     mode='train')
    process_oov_file(train_file, os.path.join(output_dir, 'train_mlm.tsv'), normal_vocab, idmap,
                     mode='train')
    process_oov_file(test_file, os.path.join(output_dir, 'test_mlm.tsv'), normal_vocab, idmap,
                     mode='test')


if __name__ == '__main__':
    fire.Fire(process_oov_words)
