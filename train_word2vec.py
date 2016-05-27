#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 10/05/2016
#    Usage: Train word2vec and jieba cut
#
############################################


from gensim.models import Word2Vec
import cPickle
import jieba


pid_p_r = cPickle.load(open("pid_p_r.pkl", "rb"))

sentences = []

for pid in pid_p_r:
    assert len(pid_p_r[pid]) == 2
    sentences.append(jieba.lcut(pid_p_r[pid][0]))
    for j in pid_p_r[pid][1]:
        sentences.append(jieba.lcut(j))


print "sentences construct done!"

print "start training ..."

model = Word2Vec(sentences, size=200, window=5, min_count=1, workers=4)

print "Word2Vec done ..."

print "save model"

model.save("model0510")