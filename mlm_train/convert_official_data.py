# -*- coding: utf-8 -*-
# @Time    : 2021/8/5
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : test.py.py
import json
import fire
import pandas as pd
from tqdm import tqdm


def unlabeled(unlabel_file):
    unlabel_data=[]
    with open(unlabel_file,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            d=json.loads(line)
            title=d['title']
            content=d['content']
            unlabel_data.append(title)
            unlabel_data.append(content)
    return unlabel_data

def main(unlabel_data='data/ori_data/sample_unlabel_data_10000.json',
         ori_train_file='data/ori_data/datagrand_2021_train.csv',
         ori_test_file='data/ori_data/datagrand_2021_test.csv',
         train_out_file='data/mlm_data/train.tsv',
         test_out_file='data/mlm_data/test.tsv',
         train_mode='unlabel,train,test',
         test_mode=''
        ):

    unlabel_data=unlabeled(unlabel_data)

    ori_train_data=[]
    ori_test_data=[]
    ori_train=pd.read_csv(ori_train_file)
    ori_test=pd.read_csv(ori_test_file)
    for index in ori_train.index:
        sentence=ori_train.loc[index,'text']+ ' 。 '
        ori_train_data.append(sentence)
    for index in ori_test.index:
        sentence=ori_train.loc[index,'text']+ ' 。 '
        ori_test_data.append(sentence)
    train_data=[]
    test_data=[]
    if 'unlabel' in train_mode:
        train_data+=unlabel_data
    if 'train' in train_mode:
        train_data+=train_data
    if 'test' in train_mode:
        train_data+=test_data

    if 'unlabel' in test_mode:
        test_data+=unlabel_data
    if 'train' in train_mode:
        test_data+=train_data
    if 'test' in train_mode:
        test_data+=test_data

    with open(train_out_file, 'w', encoding='utf-8') as f:
        for data in tqdm(train_data):
            for word in data.split():
                f.write(word + ' ')
                if not word.isdigit():
                    f.write('\n')

    with open(test_out_file,'w',encoding='utf-8') as f:
        for data in tqdm(test_data):
            for word in data.split():
                f.write(word + ' ')
                if not word.isdigit():
                    f.write('\n')


if __name__ == '__main__':
    fire.Fire(main)
