import os

import gensim
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pandas as pd
import jieba
import jieba.posseg as pseg  # 词性标注
import logging
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Activation, Softmax
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from collections import Counter


sentences = []  # 评论
labels = []  # 态度
angels = []  # 角度
w2v_model_name = os.path.join('word2vec.model')

# def read_data(data_path):
#     senlist = []
#     labellist = []
#     with open(data_path, "r",encoding='gb2312',errors='ignore') as f:
#          for data in  f.readlines():
#             data = data.strip()
#             print(data)
#             sen = data.split("\t")[1]
#             print(sen)
#             label = data.split("\t")[3]
#             if sen != "" and (label =="0" or label=="1" or label=="2" ) :
#                 senlist.append(sen)
#                 labellist.append(label)
#             else:
#                 pass
#     assert(len(senlist) == len(labellist))
#     return senlist ,labellist


data = pd.read_excel("data/merge.xlsx")

# 直接读入整个数据
print(data)

# 一些操作数据方法
# http://baijiahao.baidu.com/s?id=1655579196096844394&wfr=spider&for=pc

senlist1 = pd.read_excel("data/merge.xlsx", usecols=[1])  # 评论具体
labellist1 = pd.read_excel("data/merge.xlsx", usecols=[3])  # 态度标签
angellist = pd.read_excel("data/merge.xlsx", usecols=[4])  # 角度标签

print(senlist1)
print(labellist1)
print(angellist)

for i in range(len(senlist1)):
    # print(senlist1.iloc[i].get("cmt_cnt"))
    sentences.append(senlist1.iloc[i].get("cmt_cnt"))

for j in range(len(labellist1)):
    # print(labellist1.iloc[j].get("type"))
    labels.append(labellist1.iloc[j].get("type"))


# 统计角度标签的所有种类：
def sumAngels():
    for i in range(len(angellist)):

        # if len(angels) == 4:
        #     break
        tempAngel = angellist.iloc[i].get("label")
        tempAngel = str(tempAngel)
        # 遍历标签集
        tempAngels = tempAngel.split(",")
        for angel in tempAngels:
            if angel == 'nan':
                break
            count = 0
            if len(angels) == 0:
                angels.append(angel)
                continue
            for j in angels:
                if angel.strip() != j.strip():
                    count += 1
            # print(count)
            # print(angels)
            if count == len(angels):
                angels.append(angel)

    print(angels)


# 这一步得到了所有出现的评论角度，操作的结果对象为angels数组
sumAngels()


# print(sentences)
# print(labels)

# 上面是数据读入

# 构建角度标签和词语的词典
# words为评论分词结果，angels为所有角度标签集合

# 由值确定键
def search_key(dic, value):
    for key in dic:
        for valueList in dic[key]:
            if value in valueList:
                return key


# 建立词典,先添加查找的关键词

wordDic = {
    '外观': ['外观', '外形'],
    '充电': ['充电'],
    '拍照': ['拍照', '相机', '照片'],
    '手感': ['手感', '握', '摸'],
    '屏幕': ['屏幕', '显示', '刷新率'],
    '音效': ['音效', '音质'],
    '运行': ['运行']
}


def create_wordDic(resLb, angels):
    # 在这里实现词典构建

    # resLb不能按index查找,转换为list
    word = []
    wordLb = []
    for wd in resLb:
        if wd.flag != 'x':
            word.append(wd.word)
            wordLb.append(wd.flag)

    # '很漂亮'变成形容词……
    for index in range(len(word)):
        if word[index] == '很漂亮':
            wordLb[index] == 'a'

    # 抽取关键词
    allKey = []
    for key in wordDic:
        for keyList in wordDic[key]:
            allKey.append(keyList)

    # 所有word中定位到关键词,在后续范围内到下一个关键词之前找不在字典中的形容词加入对应角度的字典
    for i in range(len(word)):
        if word[i] in allKey:
            # print('当前关键字：' + word[i])
            for j in range(i + 1, len(word)):
                if word[j] in allKey:
                    # print('现在的词已经是另一个关键字：' + word[j])
                    break
                if wordLb[j] == 'a' and word[j] not in wordDic[search_key(wordDic, word[i])]:
                    # print('现在的词可以加入到字典中：' + word[j])
                    wordDic[search_key(wordDic, word[i])].append(word[j])
                    break

    with open("data/angels.txt", "w", encoding="utf-8") as f:
        for key in wordDic:
            print(key, ':', wordDic[key], file=f)
            print()
    f.close()
    print(wordDic)

    allKey.clear()
    for key in wordDic:
        allKey.append('/')
        for keyList in wordDic[key]:
            allKey.append(keyList)
    print(allKey)

    count = Counter(allKey)
    temp = []
    for key in count:
        if count[key] == 1:
            temp.append(key)
        if key == '/':
            temp.append(key)

    print(temp)


def train_word2vec(sentences, save_path):
    sentences_seg = []
    sen_str = "\n".join(sentences)
    # 分词
    res = jieba.lcut(sen_str)

    # 词性标注
    resLb = pseg.cut(sen_str)
    # 如果想知道某个词的词性，只需要 word.flag就可以,至于每个flag是什么意思，可以看这里
    # https://blog.csdn.net/huludan/article/details/52727298  ——jieba分词词性标注含义
    # for word in resLb:
    #     print(word.flag)
    # 那么有了词语和词性，以及标签，就可以构造词典了，在这里调用自写函数create_wordDic(),传入resLb和angels
    create_wordDic(resLb, angels)

    print(res)
    seg_str = " ".join(res)
    sen_list = seg_str.split("\n")

    print(sen_list)  # sen_list是一个句子数组，句子已经被拆成了词语

    for i in sen_list:
        sentences_seg.append(i.split())

    print(sentences_seg)  # 所有句子的集合，每一个元素是词语的集合.其实就是分词结果
    # input()

    with open("data/words.txt", "w", encoding="utf-8") as f:
        # for word in w2id.keys():
        for sen in sentences_seg:
            print(sen, file=f)
            print()
    f.close()
    # print(sentences_seg)  # 分词结果

    print("开始训练词向量")
    #     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec(sentences_seg,
                     vector_size=100,  # 词向量维度
                     min_count=5,  # 词频阈值
                     window=5)  # 窗口大小
    model.save(save_path)

    # 后续可以直接导入,还没研究出来怎么load
    # model = Word2Vec(w2v_model_name)
    return model


def load_model(word_model):
    # model = Word2Vec(vector_size=100,  # 词向量维度
    #                  min_count=5,  # 词频阈值
    #                  window=5)  # 窗口大小
    model = gensim.models.word2vec.Word2Vec.load(word_model)
    return model


model = train_word2vec(sentences, 'word2vec.model')


# model = load_model("word2vec.model")


def generate_id2wec(word_model):
    gensim_dict = Dictionary()
    # gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    gensim_dict.doc2bow(model.wv.index_to_key, allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号

    # create_wordDic的调用位置

    with open("data/keys.txt", "w", encoding="utf-8") as f:
        for word in w2id.keys():
            print(word, file=f)
            print()
    f.close()
    # print(w2id.keys())
    # for word in w2id.keys():
    #     print(model.predict_output_word(word))
    # print()
    w2vec = {word: model.wv[word] for word in w2id.keys()}  # 词语的词向量
    # print(w2vec)
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id, embedding_weights


def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    sentences_array = []
    for sen in senlist:
        new_sen = [w2index.get(word, 0) for word in sen]  # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)


def prepare_data(w2id, sentences, labels, max_len=200):
    X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train), np.array(X_val), np_utils.to_categorical(y_val)


# 获取词向量矩阵
w2id, embedding_weights = generate_id2wec(model)

# 数据处理结束
x_train, y_trian, x_val, y_val = prepare_data(w2id, sentences, labels, 200)


# 封装模型和方法
class Sentiment:
    def __init__(self, w2id, embedding_weights, Embedding_dim, maxlen, labels_category):
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # input dim(140,100)
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_test, y_test, n_epoch=5):
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                       validation_data=(X_test, y_test))
        self.model.save('sentiment.h5')

    def predict(self, model_path, new_sen):
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)

    def load_model(self, modelPath):
        self.model.load_weights(modelPath)

# # 实例化
# senti = Sentiment(w2id,embedding_weights,100,200,3)
#
# # 模型训练
# senti.train(x_train,y_trian, x_val ,y_val,1)

# senti = Sentiment(w2id,embedding_weights,100,200,3)
# senti.load_model("sentiment.h5")


# 模型预测
# label_dic = {0:"消极的",1:"积极的",2:"中性的"}
# while(1):
#     sen_new = input("来个评论兄弟:")
#     pre = senti.predict("./sentiment.h5", sen_new)
#     print("'{}'的情感是:\n{}".format(sen_new, label_dic.get(pre)))
