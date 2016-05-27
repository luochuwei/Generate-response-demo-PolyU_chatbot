#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 10/05/2016
#    Usage: construct index
#
############################################

import cPickle
import os
import jieba
from whoosh.analysis import Tokenizer,Token 
from whoosh.compat import text_type
from whoosh import fields
from whoosh import index
from whoosh.fields import *
from whoosh import scoring
from whoosh import qparser

pid_p_r = cPickle.load(open(r'pid_p_r.pkl', 'rb'))


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






        
analyzer = ChineseAnalyzer()

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer)) 

if not os.path.exists("index"):
    os.mkdir("index")

ix = index.create_in("index",schema)
ix = index.open_dir("index")

writer = ix.writer()

for pid in xrange(len(pid_p_r)):
    writer.add_document(title = str(pid).decode("utf-8"), path = u"/"+str(pid).decode("utf-8"), content = pid_p_r[pid][0].decode("utf-8"))

writer.commit()

def find(text):
    og = qparser.OrGroup.factory(0.9)
    parser = qparser.QueryParser("content", schema, group=og)
    with ix.searcher() as searcher:
        # q = parser.parse(u"今天天气不错")
        q = parser.parse(text.decode("utf-8"))
        results = searcher.search(q)
        if 0 != len(results):
            for hit in results:
                print hit.highlights("content")

    return results
