# -*- coding: utf-8 -*-
import pickle
from keras.models import load_model
from python_sample.predictor.data_processing import data_process
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

localpath = os.path.dirname(__file__)

class Predictor(object):
    # num_words=50000
    # max_len=500
    def __init__(self, num_words=50000, max_len=400,
                 # path_tokenizer=os.path.join(localpath, 'model/tokenizer_50000_cleaned.pkl'),
                 path_tokenizer=os.path.join(localpath, 'model/tokenizer_40000.pkl'),
                 path_accusation=None,
                 path_relative_articles=os.path.join(localpath,
                                                     'model/textcnn_accusation_token_40000_pad_400_filter_512_hidden_1000_epoch_2_accu_80_f1_80.h5')):
                 # path_relative_articles=os.path.join(localpath,
                 #                                     'model/textcnn_accusation_token_40000_pad_400_filter_512_hidden_1000_epoch_2_accu_80_f1_80.h5')):
        self.num_words = num_words
        self.max_len = max_len
        self.path_accusation = path_accusation
        self.path_relative_articles = path_relative_articles
        self.batch_size = 500
        self.content_process = data_process() #实例化data_process类
        self.path_tokenizer = path_tokenizer
        self.model_relative_articles = load_model(path_relative_articles)

    def predict(self, content):
        content_process = self.content_process
        content_seg = content_process.segmentation(content, cut=True, word_len=2, replace_money_value=True,
                                                   stopword=True)

        with open(self.path_tokenizer, mode='rb') as f:
            tokenizer = pickle.load(f)

        content_process.text2num(content_seg, tokenizer=tokenizer)
        content_seg_num_sequence = content_process.num_sequence
        content_fact_pad_seq = pad_sequences(content_seg_num_sequence, maxlen=self.max_len, padding='post')
        content_fact_pad_seq = np.array(content_fact_pad_seq)

        model_relative_articles = self.model_relative_articles
        relative_articles = model_relative_articles.predict(content_fact_pad_seq)

        def transform(x):
            n = len(x)
            x_return = np.arange(1, n + 1)[x > 0.5].tolist()
            if len(x_return) == 0:
                x_return = np.arange(1, n + 1)[x == x.max()].tolist()
            return x_return

        result = []
        for i in range(0, len(content)):
            result.append({
                "articles": [None],
                "imprisonment": 0,
                "accusation": transform(relative_articles[i])
            })
        return result

if __name__ == '__main__':
    content = ['荣成市人民检察院指控:\n(一)行贿事实\n2014年3月23日,被告人刘2某向文登市公安局报案称夏某职务侵占威海海地苑房地产开发有限公司(后更名为威海恒峰房地产开发有限公司)资金600余万元。2014年4月23日,文登市公安局对夏某涉嫌职务侵占罪一案立案侦查。2014年5月15日,夏某妻子徐某向文登市公安局交纳案款281万。\n2014年4、5月份的一天,被告人刘2某为谋取不正当利益,在文登市公安局副政委吴某办公室内送给吴某现金人民币5万元。\n2014年5、6月份的一天,被告人刘2某为谋取不正当利益,在文登市华夏良子足疗店门口送给吴某现金人民币10万元。\n(二)对单位行贿事实\n2014年6月12日,被告人刘2某为让文登市公安局早日将夏某职务侵占一案281万案款发还给自己,在文登市向文登市公安局行贿25万元人民币。\n被告人刘2某因涉嫌行贿罪被荣成市人民检察院传唤到案后,如实供述了荣成市人民检察院尚未掌握的对单位行贿的犯罪事实。\n针对指控的事实,公诉机关提供了相关证据。公诉机关认为,被告人刘2某为谋取不正当利益,给予国家工作人员以财物;为谋取不正当利益,给予国家机关以财物,其行为分别触犯了《中华人民共和国刑法》××××、××××、××××之规定,应当以行贿罪、××追究其刑事责任。\n被告人刘2某对公诉机关指控的犯罪事实供认不讳。']
    predictor = Predictor()
    m = predictor.predict(content)
    print(m)
