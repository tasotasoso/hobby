#Copyright (c) <2017> Suzuki N. All rights reserved.
#inspired by "http://www.phontron.com/teaching.php?lang=ja"
#coding:utf-8

import unigram as unigram
import math
import sys

#unigram.pyから{単語:確率}を取得する。
pro_dic=unigram.unigram_prob()

print("please input text!!!")
line = input()
best_score = [-1 for ii in range(len(line)+1)]
best_edge = [math.inf for ii in range(len(line)+1)]
best_edge[0] = None

#前向きステップ
for word_end in range(1,len(line)+1):
    best_score[word_end] = math.inf
    for word_begin in range(len(line)):
        word = line[word_begin : word_end]
        if word in list(pro_dic.keys()) or len(word) == 1 :
            if word in list(pro_dic.keys()):
                prob = pro_dic[word]
            else:
                prob = pro_dic["!Puni!"]
            my_score = best_score[word_begin] - math.log(prob,10)
            if my_score < best_score[word_end]:
                best_score[word_end] = my_score
                best_edge[word_end] = (word_begin, word_end)

#後ろ向きステップ
words = []
next_edge = best_edge[len(best_edge) -1]
while next_edge != None:
    word = line[next_edge[0]:next_edge[1]]
    words.append(word)
    next_edge = best_edge[ next_edge[0] ]
words.reverse()
print(words)
