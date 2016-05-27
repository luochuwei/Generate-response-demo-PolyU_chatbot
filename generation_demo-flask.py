#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 10/05/2016
#    Usage: Demo code
#
############################################


from flask import Flask
import cPickle
import os
import jieba
import math
import numpy as np
import warnings
from gensim.models import Word2Vec
from whoosh.analysis import Tokenizer,Token 
from whoosh.compat import text_type
from whoosh import fields
from whoosh import index
from whoosh.fields import *
from whoosh import scoring
from whoosh import qparser
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")



print "load data ..."
pid_p_r = cPickle.load(open(r'pid_p_r.pkl', 'rb'))
print "done."

print "load word2vec..."
#model中的词都是unicode编码  后面应用的时候 注意解码utf-8
model = Word2Vec.load('word2vec model/model0510')
print "done."



class ChineseTokenizer(Tokenizer):  
    def __call__(self, value, positions=False, chars=False, keeporiginal=False, removestops=True, start_pos=0, start_char=0, mode='', **kwargs):  
        assert isinstance(value, text_type), "%r is not unicode" % value  
        t = Token(positions, chars, removestops=removestops, mode=mode, **kwargs)  
        seglist=jieba.cut_for_search(value)                       #使用结巴分词库进行分词  
        for w in seglist:  
            t.original = t.text = w  
            t.boost = 1.0  
            if positions:  
                t.pos=start_pos+value.find(w)  
            if chars:  
                t.startchar=start_char+value.find(w)  
                t.endchar=start_char+value.find(w)+len(w)  
            yield t                                               #通过生成器返回每个分词的结果token

def ChineseAnalyzer():  
    return ChineseTokenizer()


print "Index initialization ..."
#初始化调用搜索        
analyzer = ChineseAnalyzer()

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer)) 

ix = index.open_dir("index")
print "done."


def find_post_dic(text, ix, schema):
    """
    返回的是一个字典 字典的key是post的编号
    对应字典的值是相似度的大小
    """
    og = qparser.OrGroup.factory(0.9)
    parser = qparser.QueryParser("content", schema, group=og)
    with ix.searcher() as searcher:
        # q = parser.parse(u"今天天气不错")
        try:
            q = parser.parse(text.decode("utf-8"))
        except Exception, e:
            print "OMG!!! 出错啦！"
            print "输入字符编码必须是utf-8"
        
        results = searcher.search(q, limit=20)
        pid_results_score_dic = {}
        for i in results:
            pid_results_score_dic[eval(i["title"])] = i.score

        # if 0 != len(results):
        #     for hit in results:
        #         print hit.highlights("content")

    return pid_results_score_dic


def tf_idf_c(word, text, response_candidate_dic, pid_results_score_dic):
    #word 是unicode  text还是utf-8
    mark_list = [u'。', u'.', u'!', u'！', u'?', u'？', u'；', u';',u'~',u'～',u'(', u')', u'（', u'）', u'-',u'+',u'=',u'、']
    if word in mark_list:
        return 0.005
    sp = text.decode('utf-8').split(' ')
    tf = sp.count(word)/float(len(set(sp)))

    n = 0
    for k in response_candidate_dic:
        if word.encode('utf-8') in response_candidate_dic[k][1]:
            n+=1
    for i,j in pid_results_score_dic.iteritems():
        if word.encode('utf-8') in pid_p_r[i][0]:
            n+=1
    if n == 0:
        n=1

    idf = math.log((len(response_candidate_dic)+len(pid_results_score_dic))/float(n) + 0.01)
    return tf*idf

def get_sentence_vec(text, response_candidate_dic, pid_results_score_dic):
    """
    text is utf-8
    """
    s_vec = np.zeros(model.layer1_size)
    for word in text.split(' '):
        word = word.decode('utf-8')
        if word in model.vocab:
            s_vec += tf_idf_c(word, text, response_candidate_dic, pid_results_score_dic)*model[word]
    return s_vec

def calculate_all(text, response_candidate_dic, pid_results_score_dic):
    text_vec = get_sentence_vec(text, response_candidate_dic, pid_results_score_dic)
    # print text_vec
    for ckey in response_candidate_dic:
        # print response_candidate_dic[ckey][0]
        # print pid_p_r[response_candidate_dic[ckey][0]]
        post_seg = ' '.join(jieba.cut(pid_p_r[response_candidate_dic[ckey][0]][0]))
        p_vec = get_sentence_vec(post_seg.encode('utf-8'), response_candidate_dic, pid_results_score_dic)
        r_vec = get_sentence_vec(response_candidate_dic[ckey][1], response_candidate_dic, pid_results_score_dic)
        # 计算 score 1 2 3
        s1 = cosine_similarity(text_vec, p_vec)
        s2 = cosine_similarity(text_vec, r_vec)
        s3 = cosine_similarity(p_vec, r_vec)

        all_score = s1+s2+s3
        response_candidate_dic[ckey][3] = float(s1)
        # print s1
        response_candidate_dic[ckey][4] = float(s2)
        response_candidate_dic[ckey][5] = float(s3)
        response_candidate_dic[ckey][6] = float(all_score)

    return response_candidate_dic

        






def get_response_candidate(text):
    pid_results_score_dic = find_post_dic(text, ix, schema) #text 必须是 utf-8
    response_candidate_dic = {}

    text = ' '.join(jieba.cut(text))
    text = text.encode('utf-8')

    for pid in pid_results_score_dic:
        for j in xrange(len(pid_p_r[pid][-1])):
            r_seg = ' '.join(jieba.cut(pid_p_r[pid][-1][j]))   #注意  这里的r_seg是unicode
            response_candidate_dic[str(pid)+"-"+str(j)] = [pid, r_seg.encode('utf-8'),pid_results_score_dic[pid], 0.0, 0.0, 0.0, 0.0]

    response_candidate_dic_final = calculate_all(text, response_candidate_dic, pid_results_score_dic)

    ranked_candidate_list = sorted(response_candidate_dic_final.iteritems(), key = lambda x:x[1][-1], reverse=True)
    # 返回的是这种[(2, [4, 5, 6]), (0, [1, 2, 3]), (1, [3, 2, 1])]

    return ranked_candidate_list




# r = get_response_candidate('今天天气不错')

# for i,j in enumerate(r):
#     print j[1][1].decode('utf-8'), " s1: ", j[1][3], " s2: ",j[1][4], " s3: ",j[1][5]
#     if i == 10:
#         break



def generate(text):
    r = get_response_candidate(text)
    for i,j in enumerate(r):
        print j[1][1].decode('utf-8'), " s1: ", j[1][3], " s2: ",j[1][4], " s3: ",j[1][5], " all_score: ", j[1][6]
        if i == 10:
            break







"""
Flask !!!!
"""

app = Flask(__name__)
@app.route("/")
def fff():
    return get_response_candidate('今天天气不错')

if __name__ == '__main__':

    app.run(app.debug = True)