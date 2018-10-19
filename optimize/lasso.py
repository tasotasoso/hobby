#Copyright (c) <2018> Suzuki N. All rights reserved.
#inspired by "スパース推定法による統計モデリング (統計学One Point)"
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston 

def readdata():
    """
    set X and y boston-dataset
    """

    boston = load_boston()
    boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
    df = (boston_pd - boston_pd.mean())/boston_pd.std()
    y = np.array(boston_pd[['CRIM']])
    del df['CRIM']
    X = np.array( df )
    return X,np.ravel(y)


def testsklearn(X,y):
    """
    sklearn test estimate
    """

    import sklearn.linear_model as lm
    model = lm.Lasso(alpha=1.0)
    model.fit(X, y)
    print(y)
    print(model.intercept_)
    print(model.coef_)

def CV(y,X,k):


def LS(X,y):
    """if you give X and y, return beta."""
    prod1 = np.dot(X.T, X)
    prod2 = np.dot(X.T, y)
    return np.dot(np.linalg.inv(prod1), prod2)


def Soft_threshold_operator(xj, y, param):
    """return the value of soft thereshould operator"""
    x =  np.dot(xj, y) / len(y)
    if x > 0:
        sign = 1
    elif x == 0:
        sign = 0
    else:
        sign = -1
    
    return sign * max( abs(x) - param, 0)


def estBeta_CD(X, y, betatmp, num, param):
    """
    estimate beta by coordinate descent method
    """
    Xtmp = np.delete(X,num,1)
    xj = X.T[num]
    betatmp = np.delete(betatmp,num) 
    r = y - np.dot(Xtmp, betatmp)
    return Soft_threshold_operator(xj, r, param)

def estBeta_ADMM(X, y, betatmp, gammatmp, utmp, num, param, rho):
    """
    estimate beta by ADMM
    """
    

    

def Lasso(X, y):
    """
    Lasso beta estimation.
    flag 0 : coordinate descent
    flag 1 : alternating direction method of multipliers
    """

    # method-flag setting!
    flag = 0

    # parameter setting!
    param = 1.
    threshould = 0.001
    maxcyc = 100

    #coordinate descent 
    if flag == 0:
        betatmp = np.ones( X.shape[1] )

        for ii in range( maxcyc ):
            beta = np.array( betatmp )
            for jj in range( X.shape[1] ):
                beta[jj] = estBeta_CD(X, y, beta, jj, param ) 
            if abs( sum( beta - betatmp ) ) < threshould:
                print(beta)
                return beta
            else:
                betatmp = np.array( beta )

        print('ERROR!! ITERATION NOT COMPLETED!!')
        return beta

    #alternating direction method of multipliers
    elif flag == 1:
        betatmp = np.ones( X.shape[1] )
        gammmatmp = np.ones( X.shape[1] )
        utmp = np.ones( X.shape[1] )
        
        for ii in range(maxcyc):
            beta, gammma, u = estBeta_ADMM(X, y, betatmp, gammatmp, utmp, ii, param)
            eps_beta = beta - betatmp
            eps_gammma = gammma - gammmatmp
            eps_u = u - utmp

            if sum(eps_beta) < eps and sum(eps_gammma) < eps and sum(eps_u) < eps:
                print beta

            else:
                betatmp = np.array(beta)
                gammmatmp = np.array(gammma)
                utmp = np.array(u)

        print('ERROR!! ITERATION NOT COMPLETED!!')
        return beta


def main():
    """
    i) readdata() -> get X and y.
    ii) set flag !
        flag 0 -> sklearn test estimate
        flag 1 -> least square estimator
        flag 2 -> lasso
    """
    # read data    
    X, y = readdata()

    # set method flag
    flag = 2

    if flag == 0:
        testsklearn(X,y)

    elif flag == 1:
        #LS
        beta = LS(X, y)

    elif flag == 2:
        beta = Lasso(X,y)


if __name__ == '__main__':
    main()