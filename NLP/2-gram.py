#Copyright (c) <2017> Suzuki N. All rights reserved.
#inspired by "http://www.phontron.com/teaching.php?lang=ja"

# -*- coding: utf-8 -*-
import numpy as np

def readfile(filename):
    import codecs
    f = codecs.open(filename,'r','utf-8')
    words = []
    word_num = []
    for line in f:
        line = line.strip("\n")
        words.append( line.split(" ")[0] )
        word_num.append( line.split(" ")[1] )
    nums = list(map(int,word_num))
    return words,nums

def calcPu(total):
    return 1 / total

def calcPml_omega(nums,total):
    return nums / total

def calc_lambda(nums):
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

def calcProb(nums):
    nums = np.array(nums)
    total = sum(nums)
    Pu = calcPu(total)
    my_lambda = 0.05
    return ( 1 - my_lambda ) * calcPml_omega(nums,total) + my_lambda * Pu


def main():
    import sys
    words, nums = readfile("01-train-input.txt")
    #my_lambda = calc_lambda(nums)
    print(calcProb(nums))

if __name__ == "__main__":
    main()
