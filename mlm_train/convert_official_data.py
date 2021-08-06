# -*- coding: utf-8 -*-
# @Time    : 2021/8/5
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : test.py.py
import json
import pandas as pd
from tqdm import tqdm


def unlabeled():
    unlabel_data=[]
    with open('data/ori_data/sample_unlabel_data_10000.json','r',encoding='utf-8') as f:
        for line in tqdm(f):
            d=json.loads(line)
            title=d['title']
            content=d['content']
            unlabel_data.append(title)
            unlabel_data.append(content)
    return unlabel_data

unlabel_data=unlabeled()
train_data=[]
test_data=[]
ori_train=pd.read_csv('data/ori_data/datagrand_2021_train.csv')
ori_test=pd.read_csv('data/ori_data/datagrand_2021_test.csv')
for index in ori_train.index:
    sentence=ori_train.loc[index,'text']+ ' 。 '
    train_data.append(sentence)
for index in ori_test.index:
    sentence=ori_train.loc[index,'text']+ ' 。 '
    test_data.append(sentence)

with open('data/mlm_data/unlabel_data.tsv','w',encoding='utf-8') as f:
    for data in tqdm(unlabel_data):
        for word in data.split():
            f.write(word + ' ')
            if not word.isdigit():
                f.write('\n')
                
with open('data/mlm_data/test.tsv','w',encoding='utf-8') as f:
    for data in tqdm(test_data):
        for word in data.split():
            f.write(word + ' ')
            if not word.isdigit():
                f.write('\n')
                
with open('data/mlm_data/train.tsv','w',encoding='utf-8') as f:
    for data in tqdm(train_data):
        for word in data.split():
            f.write(word + ' ')
            if not word.isdigit():
                f.write('\n')

