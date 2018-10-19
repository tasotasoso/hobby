#Copyright (c) <2017> Suzuki N. All rights reserved.
#inspired by "http://www.phontron.com/teaching.php?lang=ja"
# -*- coding: utf-8 -*-

import numpy as np

def re_uni_dic(filename):
#ファイルを開いて英単語を数え、{単語,数}を返す。
    import codecs
    from collections import Counter

    f = open(filename, "r")
    words = []

    for line in f:
        line = line.strip()
        line += " </s>"
        words += line.split(" ")

    counter = Counter(words)
    uni_dic = {}
    for word , num in counter.most_common():
        uni_dic[word] = num

    return uni_dic

def calcPu(total):
#未知語の確率を返す
    return 1 / total

def calcPml_omega(nums,total):
    return nums / total

def calc_lambda(nums):
#λの最適化を行う。
#現在はコメントアウトしてある。
    total = sum(nums)
    old_lambda = 0.5
    maxcyc = 5000
    nums = np.array(nums)
    Pu = calcPu(total)
    Pml_omega = calcPml_omega(nums,total)

    for ii in range(maxcyc):
        P_mlomega =  (old_lambda * Pml_omega) / (old_lambda * Pml_omega + (1 - old_lambda)*Pu)
        E_ml = sum( P_mlomega * nums )
        new_lambda = E_ml / total
        if abs(new_lambda - old_lambda) < 0.001:
            break

    return new_lambda

def calcProb(nums,total):
#λ=0.05のときのunigram-確率を返す。
    nums = np.array(nums)
    total = sum(nums)
    Pu = calcPu(total)
    my_lambda = 0.05
    return ( 1 - my_lambda ) * calcPml_omega(nums,total) + my_lambda * Pu


def unigram_prob():
    import sys
#corpusから頻度の学習を行う。
    uni_dic  = re_uni_dic("lib.txt")
    total = sum(list(uni_dic.values()))
    #my_lambda = calc_lambda(nums)
#uni_dicから確率の学習を行う。
    probs = (calcProb(list(uni_dic.values()),total))

#{単語,確率}を返す。
    pro_dic ={}
    keys = list(uni_dic.keys())
    for ii in range(len(probs)):
        pro_dic[keys[ii]] = probs[ii]
    pro_dic["!Puni!"] = calcPu(total)

    return pro_dic  
 
def main():
    data = unigram_prob()
    print(data)

if __name__ == "__main__":
    main()
