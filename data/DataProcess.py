import sys
sys.path.extend(['../../','./','../'])
import pandas as pd
from utils.Dataloader import Instance
from utils.utils import set_seed,write_pkl
from sklearn.model_selection import train_test_split
set_seed(666)
total_sentence=''
ori_train=pd.read_csv('data/ori_data/datagrand_2021_train/datagrand_2021_train.csv')
ori_test=pd.read_csv('data/ori_data/datagrand_2021_test/datagrand_2021_test.csv')
train_dev=[]
train_dev_labels=[]
for index in ori_train.index:
    idx=int(ori_train.loc[index,'id'])
    sentence=ori_train.loc[index,'text']
    total_sentence=total_sentence+sentence+' 。 '
    label=ori_train.loc[index,'label']
    labels=label.split('-')
    instance=Instance()
    instance.idx=idx
    instance.sentence=sentence.split(' ')
    instance.label_12=label
    instance.label_1=labels[0]
    instance.label_2=labels[1]
    train_dev.append(instance)
for inst in train_dev:
    train_dev_labels.append(inst.label_12)
test=[]
for index in ori_test.index:
    idx=int(ori_test.loc[index,'id'])
    sentence=ori_test.loc[index,'text']
    total_sentence=total_sentence+sentence+' 。 '
    instance=Instance()
    instance.idx=idx
    instance.sentence=sentence.split(' ')
    test.append(instance)

train,dev,_,_=train_test_split(train_dev,train_dev_labels,test_size=0.2,random_state=666,stratify=train_dev_labels)
print('train dev test num:{}, {}, {}'.format(len(train),len(dev),len(test)))
write_pkl(train,'data/processed/train.pkl')
write_pkl(dev,'data/processed/dev.pkl')
write_pkl(test,'data/processed/test.pkl')
with open('data/processed/train_test.txt','w',encoding='utf-8') as f:
    for word in total_sentence.split(' '):
        f.write(word+' ')
        if not word.isdigit():
            f.write('\n')
print()
