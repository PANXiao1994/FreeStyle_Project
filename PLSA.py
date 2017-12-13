# coding:utf8
from pyspark import SparkContext
from pyspark import RDD
import numpy as np
from numpy.random import RandomState

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")



class PLSA:

    def __init__(self, data, sc, k, is_test=False, max_itr=1000, eta=1e-6):

        """
        init the algorithm

        :type data RDD
        :param data: document rdd
        :type max_itr int
        :param max_itr: maximum EM iter
        :type is_test bool
        :param is_test: test or not,if yes, rd = RandomState(1)，otherwise rd = RandomState()
        :type sc SparkContext
        :param sc: spark context
        :type k int
        :param k : number of theme
        :type eta float
        :param : threshold，when the changement of log likelyhood<eta, stop iteration
        :return : PLSA object
        """

        self.max_itr = max_itr
        self.k = sc.broadcast(k)
        self.ori_data = data.map(lambda x: x.split(' '))
        self.data = data
        self.sc = sc
        self.eta = eta
        self.rd = sc.broadcast(RandomState(1) if is_test else RandomState())

    def train(self):
        #get the dictionary words
        self.word_dict_b = self._init_dict_()
        #transform the words in the documents into the indexes in the dictionary
        self._convert_docs_to_word_index()
        #initialization, the distribution under each theme
        self._init_probility_word_topic_()

        pre_l= self._log_likelyhood_()

        print("L(%d)=%.5f" %(0,pre_l))

        for i in range(self.max_itr):
            #update the posterior distribution
            self._E_step_()
            #maimize the lower bound
            self._M_step_()
            now_l = self._log_likelyhood_()

            improve = np.abs((pre_l-now_l)/pre_l)
            pre_l = now_l

            print("L(%d)=%.5f with %.6f%% improvement" %(i+1,now_l,improve*100))
            if improve <self.eta:
                break

    def _M_step_(self):
        """
        update: p(z=k|d),p(w|z=k)
        :return: None
        """
        k = self.k
        v = self.v

        def update_probility_of_doc_topic(doc):
            """
            update the distribution of the documents of the themes
            """
            topic_doc = doc['topic'] - doc['topic']
            words = doc['words']
            for (word_index,word) in words.items():
                topic_doc += word['count']*word['topic_word']
            topic_doc /= np.sum(topic_doc)

            return {'words':words,'topic':topic_doc}

        self.data = self.data.map(update_probility_of_doc_topic)
        """
        rdd相当于一系列操作过程的结合，且前面的操作过程嵌套在后面的操作过程里，当这个嵌套超过大约60，spark会报错，
        这里每次M step都通过cache将前面的操作执行掉
        """
        self.data.cache()

        def update_probility_word_given_topic(doc):
            """
            up date the distribution of the words of the themes
            """
            probility_word_given_topic = np.matrix(np.zeros((k.value,v.value)))

            words = doc['words']
            for (word_index,word) in words.items():
                probility_word_given_topic[:,word_index] += np.matrix(word['count']*word['topic_word']).T

            return probility_word_given_topic

        probility_word_given_topic = self.data.map(update_probility_word_given_topic).sum()
        probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1))

        #使每个主题下单词概率和为1
        probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

        self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

    def _E_step_(self):
        """
        update the latent viariable:  p(z|w,d)-给定文章，和单词后，该单词的主题分布
        :return: None
        """
        probility_word_given_topic = self.probility_word_given_topic
        k = self.k

        def update_probility_of_word_topic_given_word(doc):
            topic_doc = doc['topic']
            words = doc['words']

            for (word_index,word) in words.items():
                topic_word = word['topic_word']
                for i in range(k.value):
                    topic_word[i] = probility_word_given_topic.value[i,word_index]*topic_doc[i]
                #使该单词各主题分布概率和为1
                topic_word /= np.sum(topic_word)
                word['topic_word'] = topic_word # added
            return {'words':words,'topic':topic_doc}

        self.data = self.data.map(update_probility_of_word_topic_given_word)

    def  _init_probility_word_topic_(self):
        """
        init p(w|z=k)
        :return: None
        """
        #dict length(words in dict)
        m = self.v.value

        probility_word_given_topic = self.rd.value.uniform(0,1,(self.k.value,m))
        probility_word_given_topic_row_sum = np.matrix(np.sum(probility_word_given_topic,axis=1)).T

        #使每个主题下单词概率和为1
        probility_word_given_topic = np.divide(probility_word_given_topic,probility_word_given_topic_row_sum)

        self.probility_word_given_topic = self.sc.broadcast(probility_word_given_topic)

    def _convert_docs_to_word_index(self):

        word_dict_b = self.word_dict_b
        k = self.k
        rd = self.rd
        '''
        I wonder is there a better way to execute function with broadcast varible
        '''
        def _word_count_doc_(doc):
            print(doc)
            wordcount ={}
            word_dict = word_dict_b.value
            for word in doc:
                if word_dict[word] in wordcount:
                    wordcount[word_dict[word]]['count'] += 1
                else:
                    #first one is the number of word occurance
                    #second one is p(z=k|w,d)
                    wordcount[word_dict[word]] = {'count':1,'topic_word': rd.value.uniform(0,1,k.value)}

            topics = rd.value.uniform(0, 1, k.value)
            topics = topics/np.sum(topics)
            return {'words':wordcount,'topic':topics}
        self.data = self.ori_data.map(_word_count_doc_)

    def _init_dict_(self):
        """
        init word dict of the documents,
        and broadcast it
        :return: None
        """
        words = self.ori_data.flatMap(lambda d: d).distinct().collect()
        word_dict = {w: i for w, i in zip(words, range(len(words)))}
        self.v = self.sc.broadcast(len(word_dict))
        return self.sc.broadcast(word_dict)

    def _log_likelyhood_(self):
        
        probility_word_given_topic = self.probility_word_given_topic
        k = self.k
        def likelyhood(doc):
            print("succ")
            l = 0.0
            topic_doc = doc['topic']
            words = doc['words']
            for (word_index,word) in words.items():
                print(word)
                l += word['count']*np.log(np.matrix(topic_doc)*probility_word_given_topic.value[:,word_index])
            return l
        return self.data.map(likelyhood).sum()



    def save(self,f_word_given_topic,f_doc_topic):
        """
        保存模型结果 TODO 添加分布式保存结果
        :param f_word_given_topic: 文件路径，用于给定主题下词汇分布
        :param f_doc_topic: 文件路径，用于保存文档的主题分布
        :return:
        """
        doc_topic = self.data.map(lambda x:' '.join([str(q) for q in x['topic'].tolist()])).collect()
        probility_word_given_topic = self.probility_word_given_topic.value

        word_dict = self.word_dict_b.value
        word_given_topic = []

        for w,i in word_dict.items():
            word_given_topic.append('%s %s' %(w,' '.join([str(q[0]) for q in probility_word_given_topic[:,i].tolist()])))

        f1 = open (f_word_given_topic, 'w')

        for line in word_given_topic:
            f1.write(line)
            f1.write('\n')
        f1.close()

        f2 = open (f_doc_topic, 'w')

        for line in doc_topic:
            f2.write(line)
            f2.write('\n')
        f2.close()