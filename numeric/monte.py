#Copyright (c) <2017> Suzuki N. All rights reserved.
#for Training session at O.U.Library 
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def sampling_data(sample_size):
    #データ点の生成
    x = np.random.rand(sample_size)
    y = np.random.rand(sample_size )
    return [x,y]

def calc_dist(x,y):
    #piの計算
    if np.sqrt(x**2 + y**2) > 1:
       return 0
    else: 
       return 1

def calc_pi(samples):
    sample_size = len(samples[0])
    result = [calc_dist(samples[0][ii],samples[1][ii]) for ii in range( sample_size)]
    return np.sum(result)/sample_size

def output_pi(pi):
    print(pi * 4)

def plot_MonteCarlo(samples):
    #イメージの描写
    x = samples[0]
    y = samples[1]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    circle = plt.Circle((0,0),1,alpha=0.3,fc="#770000")
    ax.add_patch(circle)

    ax.scatter(x,y)
    ax.set_title('Prediction of pi by Monte Carlo method')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.axis((0,1,0,1))
    ax.set_aspect(1)

    plt.show()


def main():
    sample_size = input()
    samples = sampling_data(int(sample_size))
    pi_quarter = calc_pi(samples)
    output_pi(pi_quarter)
    plot_MonteCarlo(samples)

if __name__ == "__main__":
    main()

