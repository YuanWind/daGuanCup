#! -*- coding:utf-8 -*-
# 句子对分类任务，脱敏数据
# 比赛链接：https://tianchi.aliyun.com/competition/entrance/531851

import json
from collections import defaultdict

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from bert4keras.backend import keras, K,search_layer
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm
from bert4keras.layers import *
from keras.utils import np_utils
from model import tokenization

from sklearn.model_selection import StratifiedKFold
import json
import numpy as np
from sklearn.metrics import f1_score
from bert4keras.backend import keras, K
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm
from bert4keras.layers import *
import os
min_count = 3
maxlen = 256
batch_size = 32

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True                                #按需分配显存
keras.backend.set_session(tf.Session(config=config))

config_path = '../models/roberta_zh_l12/bert_config.json'
checkpoint_path = '../models/result/best_model_99.weights'
dict_path = '../models/roberta_zh_l12/vocab.txt'


label2id = {}
id2lable = {}

def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for id,l in enumerate(f):
            if id == 0:
                continue
            l = l.strip().split(',')
            if len(l) == 3:
                a, b, c = l[0], l[1], l[2]
            else:
               continue
            # a = [int(i) for i in a.split(' ')]
            b = [i for i in b.split(' ')]
            if c not in label2id.keys():
                label2id[c] = len(label2id)
                id2lable[label2id[c]] = c
            # truncate_sequences(maxlen, -1, a, b)
            D.append((b[0:maxlen],c，a))
    return D




# 加载数据集
data = load_data(
    '../data/datagrand_2021_train.csv'
)

cates = len(label2id.keys())

import random
random.shuffle(data)

train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data(
    '../data/datagrand_2021_test.csv'
)

tokenizer = tokenization.FullTokenizer(
    vocab_file=dict_path,
    do_lower_case=True)



def sample_convert(text1, random=False):
    """转换为MLM格式
    """

    tokens = tokenizer.tokenize(" ".join(text1[0]))
    text1_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])

    token_ids = [2] + text1_ids + [3]
    segment_ids = [0] * len(token_ids)
    output_ids = [label2id[text1[1]]]
    return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, text1 in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    #checkpoint_path=checkpoint_path,
    with_mlm=True,
    return_keras_model= False
    #keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)]
)
bert_k = bert
model1 = bert.model
model1.load_weights(checkpoint_path)

input_mask = Masking(mask_value=0)(model1.inputs[0])
input_mask = Lambda(
    lambda x: K.cast(K.any(x, axis=-1, keepdims=True), 'float32')
)(input_mask)

output1 = model1.get_layer(name='Transformer-11-FeedForward-Norm').output

# output1 = Multiply()([input_mask, output1])
#
# output1_S = MultiHeadAttention(heads=4, head_size=64)([output1, output1, output1])
#
# output1 = Add()([output1_S, output1])
#
output1 = Lambda(lambda inp: K.mean(inp, axis=1))(output1)

output1 = Dense(
    units=cates, activation='softmax', kernel_initializer=bert_k.initializer
)(output1)

model = keras.models.Model(model1.input, output1)


model.summary()
#
# 优化器
optimizer = extend_with_weight_decay(Adam)
# if which_optimizer == 'lamb':
#     optimizer = extend_with_layer_adaptation(optimizer)
optimizer = extend_with_piecewise_linear_lr(optimizer)
optimizer_params = {
    'learning_rate': 3e-5,
    'weight_decay_rate': 0.01,
}

optimizer = optimizer(**optimizer_params)

model.compile(
    # loss=keras.losses.sparse_categorical_crossentropy(from_logits=False),
    # loss = 'binary_crossentropy',
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=optimizer,  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
)
def adversarial_training(model, embedding_name, epsilon=1.):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数
adversarial_training(model, 'Embedding-Token', epsilon=0.5)


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    """线下评测函数
    """
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        Y_pred.extend(np.argmax(y_pred,axis=-1))
        Y_true.extend(y_true)
    return f1_score(Y_true, Y_pred,average='macro')


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights('best_model.weights')
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )

def predict_to_file(out_file):
    """预测结果到文件
    """
    F = open(out_file, 'w')
    for x_true, y_true in tqdm(test_generator):
        y_pred = model.predict(x_true)
        y_pred = np.argmax(y_pred,axis=-1)
        for p in y_pred:
            F.write('%f\n' % p)
    F.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

else:
    model.load_weights('best_model.weights')
