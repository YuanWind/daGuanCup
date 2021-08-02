import sys
sys.path.extend(["../../","../","./"])
import logging
from utils.utils import load_pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
     
def analysis(data):
    data_json={'dialog_id':[],'sentence':[],'sentence_len':[]}
    dialog_lens={}
    
    for inst in data:
        dialog_id=inst.dialog_id
        speaker=inst.speaker
        sentence=inst.ori_sentence
        data_json['dialog_id'].append(dialog_id)
        data_json['sentence'].append(speaker+sentence)
        data_json['sentence_len'].append(len(speaker+sentence))
        if dialog_id not in dialog_lens.keys():
            dialog_lens[dialog_id]=[len(speaker+sentence)]
        else:
            dialog_lens[dialog_id].append(len(speaker+sentence))
    data2_json={'dialog_id':[],'每句话长度':[],'对话轮数':[],'对话总字数':[],'句子最大长度':[],'句子平均长度':[]}
    for idx,lens in dialog_lens.items():
        data2_json['dialog_id'].append(idx)
        data2_json['每句话长度'].append(lens)
        data2_json['句子最大长度'].append(max(lens))
        data2_json['对话总字数'].append(sum(lens))
        data2_json['对话轮数'].append(len(lens))
        data2_json['句子平均长度'].append(np.mean(lens))
    df1=pd.DataFrame(data_json)
    df2=pd.DataFrame(data2_json)
    res1=df1.describe()
    res2=df2.describe()
    print(res1)
    print(res2)
    len_df = df1.groupby('sentence_len').count()
    sent_length = len_df.index.tolist()
    sent_freq = len_df['sentence'].tolist()
    plt.bar(sent_length, sent_freq)
    plt.title("sent_len-freq")
    plt.xlabel("sent_len")
    plt.ylabel("freq")
    plt.savefig("./总体.jpg")
    plt.cla()
    len2_df=df2.groupby('对话总字数').count()
    sent2_length = len2_df.index.tolist()
    sent2_freq = len2_df['dialog_id'].tolist()
    max_freq= max(sent2_freq)
    plt.bar(sent2_length, sent2_freq)
    plt.xlim(0, 2000)
    plt.ylim(0,max_freq)
    plt.title("dialog_len-freq")
    plt.xlabel("dialog_sum_len")
    plt.ylabel("freq")
    plt.savefig("./对话.jpg")
    
    print()
def pos_neg_look(data):
    neg_num=0
    pos_num=0
    for inst in data:
        if len(inst.symptom_norm) > 0:
            pos_num+=1
        else:
            neg_num+=1
    return pos_num,neg_num
if __name__ == '__main__':
    train_data=load_pkl('../processed_data/phrase2_train_sentences.pkl') # list类型
    dev_data=load_pkl('../processed_data/phrase2_dev_sentences.pkl')     # list 类型
    pos_num,neg_num=pos_neg_look(train_data)
    print('训练集有症状的样本数(正样本):{}, 无症状的样本数(负样本):{}'.format(pos_num,neg_num))
    pos_num,neg_num=pos_neg_look(dev_data)
    print('验证集有症状的样本数(正样本):{}, 无症状的样本数(负样本):{}'.format(pos_num,neg_num))