import json
import jieba
import numpy as np
import re
from keras.preprocessing.text import Tokenizer

jieba.setLogLevel('WARN')


class data_process(object):
    def __init__(self):
        self.data_label = {}
        self.label_collection = {}
        self.label_type = {'law': 'relevant_articles', 'accu': 'accusation'}
        self.decomposition = {}

    def get_data(self, file_path=None):

        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            file = f.readlines()
            for content in file:
                data.append(json.loads(content))
        self.data_path = file_path
        self.data = data

    def decompose_data(self, name=''):

        try:
            decomposition = [i[name] for i in self.data]
        except:
            decomposition = [i['meta'][self.label_type[name]] for i in self.data]
            if name == 'accu':
                decomposition = [[re.sub('[\[\]]', '', label) for label in i] for i in decomposition]
        self.decomposition.update({name: decomposition})

    def replace_money_value(self, string):

        moeny_list = [1, 2, 5, 7, 10, 20, 30, 50, 100, 200, 500, 800, 1000, 2000, 5000, 7000, 10000, 20000, 50000,
                      80000, 100000, 200000, 500000, 1000000, 3000000, 5000000, 1000000000]
        double_patten = r'\d+\.\d+'
        int_patten = r'[\u4e00-\u9fa5,，.。；;]\d+[元块万千百十余，,。.;；]'
        doubles = re.findall(double_patten, string)
        ints = re.findall(int_patten, string)
        ints = [a[1:-1] for a in ints]
        # print(doubles+ints)
        sub_value = 0
        for value in (doubles + ints):
            for money in moeny_list:
                if money >= float(value):
                    sub_value = money
                    break
            string = re.sub(str(value), str(sub_value), string)
        return string

    def data_cleaning(self, lines):

        # 去掉日期的数字
        date_list = [str(i) for i in np.array(range(0, 31)) + 1]
        year_list = [str(i) for i in np.array(range(2000, 2018)) + 1]
        stopword_list = ['某某', '某.+', r'.+省\b', r'.+市\b', r'.+某\b', r'.+市\b', r'.+区\b', '被告人',r'xx\b',
                         '有限公司', '分公司', r'.+乡\b', '人民检察院', '指控', r'.+县\b', '时许', '下午', '上午']
        stopword_pattern = '|'.join(stopword_list)
        lines1 = []
        for sentence in lines:
            sentence1 = []
            for word in sentence:
                if word in date_list or word in year_list:
                    continue
                elif re.findall(stopword_pattern, word) != []:
                    continue
                else:
                    sentence1.append(word)
            lines1.append(sentence1)
        return lines1

    def segmentation(self, list=None, cut=True, word_len=1, path=None, replace_money_value=False, stopword=False):

        if cut == False:
            seg_list = [[word for word in text if len(word) >= word_len] for text in list]
        else:
            if replace_money_value == True:
                seg_list = [[word for word in jieba.lcut(self.replace_money_value(text)) if len(word) >= word_len] for
                            text in list]
            else:
                seg_list = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in list]

        if stopword == True:
            seg_list = self.data_cleaning(seg_list)

        if path != None:
            with open(path, 'w') as f:
                json.dump(seg_list, f)
        return seg_list

    def get_label_collection(self, label_type=None):

        if label_type == None:
            labels = self.label_type
        else:
            labels = label_type.split('_')

        for label in labels:
            with open('../good/%s.txt' % label, 'r', encoding='utf-8') as f:
                label_set = f.read().split()
            self.label_collection.update({label: label_set})

    def text2num(self, text_list=None, tokenizer=None, num_words=40000):

        list_length = len(text_list)
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=num_words)
            if list_length > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < list_length:
                tokenizer.fit_on_texts(texts=text_list[n:n + 10000])
                n += 10000
                if n < list_length:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % list_length)
            self.tokenizer = tokenizer

            # 全部转为数字序列
        num_sequence = tokenizer.texts_to_sequences(texts=text_list)
        self.num_sequence = num_sequence
        print('finish texts to sequences')

        del text_list

    def one_hot(self, label, label_set):

        label = [str(i) for i in label]
        one_hot = (np.in1d(np.array(label_set), label) + 0).reshape(1, -1)
        return one_hot

    def transform_label(self, labels=None, label_type='law'):

        label_one_hot = np.concatenate([self.one_hot(label, self.label_collection[label_type]) for label in labels])
        return label_one_hot
