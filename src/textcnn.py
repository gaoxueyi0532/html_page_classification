# usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import json
import logging
import numpy as np
import tensorflow as tf
from time import sleep

import gensim
from gensim.models import Word2Vec
from gensim import models

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling1D
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras import optimizers
from keras import regularizers
from keras import backend as K

logger = logging.getLogger()
logger.setLevel(logging.INFO)


'''
    将训练数据中的单词列表转换为对应在词典中的索引列表
    如：石油将被编码为100, 表示石油在词典中的索引是100, 未发现的词索引标记为0
    length: 最大的标题长度, 默认为16, 长度不足16尾部添0
'''
def encode_data(train_data, word_index_map, length=16):
    encode_data = []
    for item in train_data:
        encode_item = []
        for token in item:
            index = word_index_map.get(token, 0)
            encode_item.append(index)
        ln = len(encode_item)
        if ln < length:
            diff = length - ln
            encode_item.extend([0] * diff)
        encode_data.append(encode_item[0 : length])
    return encode_data

''' this class provide word embedding, word_index and word_vector '''
class WordEmbed(object):
    def __init__(self):
        self.word_index = {" " : 0}
        self.word_vector = {}
        self.word_embedding = None
        self.vocab_size = 0
        self.embed_size = 0

    def prepare(self):
        # 加载word2vec模型
        mpath = "../model/binary_embed.model"
        model = models.KeyedVectors.load_word2vec_format(mpath, binary=True)
        self.embed_size = model.vector_size

        # 构造包含所有词的词典
        token_list = [word for word, vocab in model.wv.vocab.items()]
        print("token list size: %d" % len(token_list))
        self.vocab_size = len(token_list)

        self.word_embedding = np.zeros((len(token_list) + 1, model.vector_size))

        # build word embedding matrix
        for i in range(len(token_list)):
            word = token_list[i]
            # <word, index>
            self.word_index[word] = i + 1
            # <word, vector>
            self.word_vector[word] = model.wv[word]
            # row num is length of token_list
            # colnmn num is model.vector_size
            self.word_embedding[i + 1] = model.wv[word]

        # output embedding shape
        print("Word embedding shape: %s" % str(self.word_embedding.shape))

    def get_word_embedding(self):
        return self.word_embedding

    def get_word_index(self):
        return self.word_index

    def search_word_index(self, word):
        return self.word_index.get(word, 0)

    def get_word_vector(self):
        return self.word_vector

    def get_vocab_size(self):
        return self.vocab_size

    def get_embed_size(self):
        return self.embed_size


class TextCNN(object):
    def __init__(self):
        self.model = None

    def prepare(self):
        if os.path.exists("../model/textcnn_v6.model"):
            print("load textcnn model...")
            self.model = load_model("../model/textcnn_v6.model")
            weights = np.array(self.model.get_weights())
            for w in weights:
                print("layer shape: %s" % str(w.shape))
                print(w)
            print("load finish!")
            lst = [0]*16
            self.test(lst)

    def build_model(self, length, vocab_size, embed_size, embedding):
        # version_1
        '''
        filter_size = [16, 32, 64]
        kernal_size = [2, 3, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [2, 2, 2]
        hide_layer_size = 100
        '''

        # version_2
        # 1. change kernal initializer of Dence layer
        '''
        filter_size = [16, 32, 64]
        kernal_size = [2, 3, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [2, 2, 2]
        hide_layer_size = 100
        '''

        # version_3
        # 1. change hide_layer_size of dense1 from 100 to 50
        # 2. add kernel_regularizer=regularizers.l2(0.01) to all Dense layer
        filter_size = [16, 32, 64]
        kernal_size = [2, 3, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [2, 2, 2]
        hide_layer_size = 50

        # version_4
        # 1. add kernel_regularizer=regularizers.l2(0.05) to all Dense layer
        filter_size = [16, 32, 64]
        kernal_size = [2, 3, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [2, 2, 2]
        hide_layer_size = 50

        # version_5
        # 1. add kernel_regularizer=regularizers.l2(0.1) to all Dense layer
        filter_size = [16, 32, 64]
        kernal_size = [2, 3, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [2, 2, 2]
        hide_layer_size = 50

        # version_6
        # 1. add kernel_regularizer=regularizers.l2(0.5) to all Dense layer
        # 2. filter[2] changes to 4
        # 3. filter[1] changes to 4
        # 4. filter[0] changes to 4
        # 5, dense 2 initialization changes to he_initialize
        filter_size = [4, 8, 32]
        kernal_size = [1, 2, 4]
        drop_ratio = [0.5, 0.5, 0.5]
        pooling_size = [4, 2, 2]
        hide_layer_size = 50

        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, embed_size, weights=[embedding], trainable=False)(inputs1)
        conv1 = Conv1D(filters=filter_size[0], kernel_size=kernal_size[0], activation='relu')(embedding1)
        drop1 = Dropout(drop_ratio[0])(conv1)
        pool1 = MaxPooling1D(pool_size=pooling_size[0])(drop1)
        flat1 = Flatten()(pool1)

        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, embed_size, weights=[embedding], trainable=False)(inputs2)
        conv2 = Conv1D(filters=filter_size[1], kernel_size=kernal_size[1], activation='relu')(embedding2)
        drop2 = Dropout(drop_ratio[1])(conv2)
        pool2 = MaxPooling1D(pool_size=pooling_size[1])(drop2)
        flat2 = Flatten()(pool2)

        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, embed_size, weights=[embedding], trainable=False)(inputs3)
        conv3 = Conv1D(filters=filter_size[2], kernel_size=kernal_size[2], activation='relu')(embedding3)
        drop3 = Dropout(drop_ratio[2])(conv3)
        pool3 = MaxPooling1D(pool_size=pooling_size[2])(drop3)
        flat3 = Flatten()(pool3)

        # merge
        merged = concatenate([flat1, flat2, flat3])
        dense1 = Dense(hide_layer_size, activation='relu', \
                       kernel_initializer='he_normal', \
                       bias_initializer='he_normal', \
                       kernel_regularizer=regularizers.l2(0.5), \
                       bias_regularizer=regularizers.l2(0.1))(merged)
        outputs = Dense(1, activation='sigmoid',\
                       kernel_regularizer=regularizers.l2(0.1))(dense1)
        self.model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        # compile
        adm_opt = optimizers.Adam(lr=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=adm_opt, metrics=['accuracy'])

        # summarize
        print(self.model.summary())
        plot_model(self.model, show_shapes=True, to_file='multichannel.png')
        return self.model

    def train(self, x, y, epoch_num, batch_sz, val_data):
        #tbCallBack = TensorBoard(log_dir="./log", histogram_freq=1, write_grads=True)
        #self.model.fit(x, y, epochs=epoch_num, batch_size=batch_sz,\
        #        validation_data=val_data, verbose=1, callbacks=[tbCallBack])
        self.model.fit(x, y, epochs=epoch_num, batch_size=batch_sz, validation_data=val_data, verbose=1)

    def evaluate(self, text_x, test_y):
        self.model.fit(test_x, test_y)

    # lst is a index array, element corresponse to the word index in vocab dict
    # such as [2,4,233,1,23,0,0,0,0,9823,0,0,70,0,0,35]
    def test(self, lst, lenght=16):
        try:
            if not lst:
                logging.info("input empty!")
                return (False, 0.0)
            ln = len(lst)
            if ln < lenght:
                lst.extend([0] * (lenght - ln))
            else:
                lst = lst[0 : lenght]
            x = np.array(lst).reshape(1,lenght)
            if self.model is not None:
                prob = float(self.model.predict([x,x,x]))
                return (prob > 0.6, prob)
            else:
                logging.info("No textcnn model loaded!")
                return (False, 0.0)
        except Exception as e:
            raise Exception("textcnn exception: %s, %s" % (str(e), str(x)))


''' 加载分词后的标题token列表 '''
def load_data(file_name):
    f = open(file_name, "r", encoding="utf-8", errors='ignore')
    n = 0
    train_data = []
    max_len = 0
    try:
        line = next(f)
        while line:
            try:
                n += 1
                d = json.loads(line.strip())
                token_list = d["token_seq"].split(',')
                if len(token_list) > max_len:
                    max_len = len(token_list)
                train_data.append(token_list)
                line = next(f)
            except Exception as e:
                print("inner error: %s" % str(e))
                line = f.next()
                continue
    except Exception as e:
        print("outer error: %s" % str(e))
    finally:
        f.close()
    print("%d records in %s" % (n, file_name))
    print("max len: %d" % max_len)
    return train_data

def train(model_name):
    # load train data
    pos_x = load_data("../data/news_records.txt")
    pos_y = [1] * len(pos_x)
    neg_x = load_data("../data/other_records.txt")
    neg_y = [0] * len(neg_x)

    word_embed = WordEmbed()
    word_embed.prepare()
    word_index_map = word_embed.get_word_index()
    word_embedding = word_embed.get_word_embedding()

    x = []
    x.extend(pos_x)
    x.extend(neg_x)
    x = encode_data(x, word_index_map)
    print("encoded data item: %s" % str(x[0]))

    y = []
    y.extend(pos_y)
    y.extend(neg_y)
    p = [i for i in range(len(x))]
    for i in range(200):
        random.shuffle(p)

    # split data into train set, validata set and test set, as 0.7 : 0.2 : 0.1
    ds = len(p)
    train_x = [x[i] for i in p[0 : int(0.7*ds)]]
    train_y = [y[i] for i in p[0 : int(0.7*ds)]]
    val_x = [x[i] for i in p[int(0.7*ds)+1 : int(0.9*ds)]]
    val_y = [y[i] for i in p[int(0.7*ds)+1 : int(0.9*ds)]]
    test_x = [x[i] for i in p[int(0.9*ds)+1 : ds]]
    test_y = [y[i] for i in p[int(0.9*ds)+1 : ds]]
    print("size: %d, %d, %d" % (len(train_x), len(val_x), len(test_x)))

    # define model
    textcnn = TextCNN()
    vocab_size = word_embed.get_vocab_size()
    embed_size = word_embed.get_embed_size()
    model = textcnn.build_model(16, vocab_size+1, embed_size, word_embedding)

    # clear data for save memory
    del pos_x[:]
    del neg_x[:]

    # train model
    tx = np.array(train_x)
    vx = np.array(val_x)
    vy = np.array(val_y)
    val_data = ([vx,vx,vx], vy)
    textcnn.train([tx, tx, tx], np.array(train_y), epoch_num=5, batch_sz=4096, val_data=val_data)

    # save model
    model.save('../model/' + model_name)

    # load model
    model = load_model("../model/" + model_name)
    tx = np.array(test_x)
    ty = np.array(test_y)
    loss, acc = model.evaluate([tx,tx,tx], ty, verbose=0)
    print("test loss and acc: %s, %s" % (str(loss), str(acc)))

def test():
    # test
    embed = WordEmbed()
    embed.prepare()
    textcnn = TextCNN()
    textcnn.prepare()

    '''
    n = 0
    g = 0
    f = open("../data/other_records.txt", "r", encoding="utf-8", errors='ignore')
    try:
        line = next(f)
        while line:
            try:
                d = json.loads(line.strip())
                lst = d["token_seq"].split(',')
                g += 1
                indexs = [embed.search_word_index(w) for w in lst]
                print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
                if textcnn.test(indexs)[1] > 0.5:
                    n += 1
                line = next(f)
            except:
                line = next(f)
                continue
    except Exception as e:
        print("error: %s, %s" % str)
    finally:
        f.close()
    print("gxy: %d, %d" % (n, g))
    '''

    lst = ['第三', '流派', ' ', '金刚', '技能', '一览']
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["调整","完善","土地","收入","使用","范围","优先","支持","乡村","振兴"]
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["胸腔","按压", "如何","止痛"]
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["穴位","按压", "如何","找准","穴位"]
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["心脏","不好","常喝","药茶"]
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["快", "检查","家庭","药箱","药品", "放对"]
    indexs = [embed.search_word_index(w) for w in lst]
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ["男子","女友","看不起","结果","后悔不已"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["五月","女王","图片","海绵","可以","图片","大全","人体"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["平湖","广厦","爱尚","绿地","房子","怎么样","可以"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["平湖","龙湖","春江","天玺","房子","怎么样","可以"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["上海","立马","装修","公司","怎么样"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['返回', '中国', '金融', '信息网', '首页']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["分析", "特朗普", "性格"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ["特朗普","女儿","居然","网店"]
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['神农架', '晨雾', '缭绕', ' ', '秋色', '迷人']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['践行', '绿色', '发展', '理念', ' ', '增加', '绿化', '面积']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['一页']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['医学论坛', '肿瘤']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['医学论坛', '皮肤性病']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['肺癌', '重磅']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['肾上腺', '皮质', '功能', '异常', '伴发', '精神障碍']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['高血压', '领域', '研究进展', '2020']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['开胸', '手术', '限制性', '液体', '管理', '复合', '并发症']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['勾勾', '拇指', '挑逗', '亚洲', '女人', '只有']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['白斑']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['秦始皇', '秦直道']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['影响', '白癜风', '治疗', '几大', '因素']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['依氟', '鸟氨酸', '舒林', '治疗', '腺瘤', '息肉']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['社会保险', '个人']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))

    lst = ['黄金']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['疫情']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['会议']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['肿瘤']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['怎么样']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['装修']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))
    lst = ['水产']
    indexs = [embed.search_word_index(w) for w in lst]
    #print(indexs)
    print("%.2f, %s" % (textcnn.test(indexs)[1], str(lst)))


def main():
    # test
    #test()
    # train
    train("textcnn_v6.model")

if __name__ == '__main__':
    main()
