#Copyright (c) <2017> Suzuki N. All rights reserved.
#inspired by "http://www.phontron.com/teaching.php?lang=ja"
# -*- coding: utf-8 -*-

import MeCab
from collections import Counter

def Wordcount(text):

    def myparse(text):
        mc = MeCab.Tagger("-Owakati")
        ptext = mc.parse(text)
        words = ptext.split()
        return words

    def count(words):
        counter = Counter(words)
        for word, cnt in counter.most_common():
            print(word, cnt)

    words = myparse(text)
    count(words)

def readfile(filename):
    f = open("text.txt",'r')
    text = ""
    for line in f:
        text += line.strip()
    return text

def main():
    import sys
    f1 = open(sys.argv[1], 'r')
    text = readfile(f1)
    Wordcount(text)

if __name__ == "__main__":
    main()

