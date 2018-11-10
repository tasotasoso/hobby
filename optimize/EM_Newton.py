#Copyright (c) <2018> Suzuki N. All rights reserved.
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def inputT0():
    """
    T0を入力する。
    今回は3*3のE
    """
    return np.array([[1,0,0],[0,1,0],[0,0,1]]) 

def inputQ():
    """
    Qを入力する。 
    """
    return np.array([[ 0.13869439, -0.04564595,  0.00581838],[-0.04564595,  0.19457736, -0.00556244],[ 0.00581838, -0.00556244,  0.11172976]])

def plotCalcLog(calclog):
    """
    f(T)の更新をplotする。
    plot終了後にlogを出力する。(見にくいのでコメントアウトしときます。)
    """
    x = range(len(calclog))
    plt.plot(x, calclog)
    plt.show()
    #コメントを外すとログを出力する。#
    #print("CALCLOG:", calclog, ":CALCLOG")

def JacobianM(T, Q):
    """
    TとQからJacobian Matrixを計算する。
    """
    T_dash_inv = np.linalg.inv(T.T)
    T_inv = np.linalg.inv(T) 
    return 2*np.linalg.det( np.dot(T.T, T) )*T_dash_inv - 2 * np.dot( np.dot(T_dash_inv, Q), np.dot(T_inv, T_dash_inv) ) 

def f(T, Q):
    """
    TとQからf(T)を計算する。
    """
    T_dash_T = np.dot(T.T, T)
    return np.log( np.linalg.det(T_dash_T) ) + np.trace( np.dot( np.linalg.inv(T_dash_T), Q) )

def M_Newton(T0):
    """
    T0を初期値としたNewton法を行う。
    gradはJacobianM(T, Q)としている。
    """
    maxcyc = 1000
    calclog = []
    Q = inputQ() 
    old_f = f(T0,Q)

    for i in range(maxcyc):
        T = T0 - np.dot( np.linalg.inv( JacobianM(T0, Q) ), f(T0,Q) )
        new_f = f(T, Q)
        calclog.append(new_f)
        if abs(new_f - old_f) < 0.003:
            print("Minimization complete!!!")
            return T, calclog
        T0 = T
        old_f = new_f

    print("Minimization is not completed!!!")
    return T, calclog


def M_Decelerat_Newton(T0):
    """
    T0を初期値とした減速Newton法を行う。
    gradはJacobianM(T, Q)としている。
    """
    maxcyc = 1000
    calclog = []
    Q = inputQ() 
    old_f = f(T0,Q)

    for i in range(maxcyc):
        mu = 1
        while True:
            T = T0 - mu * np.dot( np.linalg.inv( JacobianM(T0, Q) ), f(T0,Q) )
            new_f = f(T, Q)
            if(abs(new_f) - (1-mu/4)*abs(old_f) < 0):
                break
            mu /= 2

        calclog.append(new_f)
        if abs(new_f - old_f) < 0.00001:
            print("Minimization complete!!!")
            return T, calclog
        T0 = T
        old_f = new_f

    print("Minimization is not completed!!!")
    return T, calclog


def main():
    T0 = inputT0()
    T, calclog = M_Decelerat_Newton(T0)
    print("T: ", T)
    plotCalcLog(calclog) 

if __name__ == "__main__":
    main()