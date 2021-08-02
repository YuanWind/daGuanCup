# -*- coding: utf-8 -*-
# @Time    : 2021/8/2
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : test.py
import numpy as np
import pandas

from utils.utils import load_pkl
test_data_file='data/processed/test.pkl'
test_data=load_pkl(test_data_file)
vocab=load_pkl('vocab.pkl')
predict_res=load_pkl('predict_res.pkl')
preds=predict_res[0]
pred_labels=np.argmax(preds, axis=-1)
res={'id':[],'label':[]}
for index,inst in enumerate(test_data):
    idx=inst.idx
    label_id=pred_labels[index]
    res['id'].append(idx)
    res['label'].append(vocab.id2label[label_id])
res=pandas.DataFrame(res)
res.to_csv('res.csv',index=False)
print()