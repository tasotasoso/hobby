#-*- coding: utf-8 -*-
#Copyright (c) <2017> Suzuki N. All rights reserved
#inspired "http://deeplearning.net/tutorial/" 


import theano
import theano.tensor as T
import numpy as np
import numpy.random 
import gzip
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import math

def init_params(n_in, n_out):
# パラメータの初期化
    wid = math.sqrt(6/(n_in + n_out))
    init_w1 = numpy.random.uniform( -wid, wid, (n_in, n_in//3))
    init_w2 = numpy.random.uniform( -wid, wid, (n_in//3, n_out))
    init_b1 = numpy.zeros(n_in//3)
    init_b2 = numpy.zeros(n_out)
    w1 = theano.shared(init_w1,name='w1')
    w2 = theano.shared(init_w2,name='w2')
    b1 = theano.shared(init_b1,name='b1')
    b2 = theano.shared(init_b2,name='b2')
    return [w1,w2,b1,b2]

def load_data():
# データセットの読み込み
# http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
# をダウンロードして dataset_path においておく。

    dataset_path = "./mnist.pkl.gz"
    with gzip.open(dataset_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return [train_set, valid_set, test_set]

def label_to_onehot(labels):
# http://testpy.hatenablog.com/entry/2017/01/19/232049 を参考。
#普通に内包表記で良い。

    from sklearn.preprocessing import OneHotEncoder

    # reshape
    X = np.array(labels).reshape(1, -1) 

    # transpose
    X = X.transpose()

    # encode label
    encoder = OneHotEncoder(n_values=max(X)+1)
    X = encoder.fit_transform(X).toarray()
    return X.astype(np.int)

def label_size(t_train_sets):
    return max(t_train_sets) + 1

def build_train(params, datasets):
# パラメータの初期値を設定する。
# ハイパーパラメタは適当。
    w1 = params[0]
    w2 = params[1]
    b1 = params[2]
    b2 = params[3]
    train_set_x, train_set_t = datasets
    batch_size = 100
    learning_rate = 0.5
    iternum = 1500

# 多層パーセプトロンモデル
# loss-functionは多項分布のcross-entropy
# cross-entropyの書きやすさを考えて、train_set_tはone-hot-vectorにした。
    x = T.dmatrix('x')
    h1 = T.nnet.relu( T.dot(x,w1) + b1 )
    h2 =  T.nnet.relu( T.dot(h1,w2) + b2 )
    y =  T.nnet.softmax( h2  )
    t = T.lmatrix('t')
    train_set_t = label_to_onehot(train_set_t)
    cross_entropy = T.mean(T.sum(-t * T.log(y) - (1-t) * T.log(1-y),axis=1))

# loss-functionの微分モデルの作成
    gw1 = T.grad(cost=cross_entropy, wrt=w1)
    gw2 = T.grad(cost=cross_entropy, wrt=w2)
    gb1 = T.grad(cost=cross_entropy, wrt=b1)
    gb2 = T.grad(cost=cross_entropy, wrt=b2)

# theano-functionの定義
# 入力はxとt、出力はy、updatesによりf1を呼び出すごとにSGDで更新する。
    f1 = theano.function(inputs = [x, t], outputs = y, updates=[[w1, w1 - learning_rate * gw1],[b1, b1 - learning_rate * gb1],[w2, w2 - learning_rate * gw2],[b2, b2 - learning_rate * gb2]])
    f_cost = theano.function([x,t],cross_entropy) 
    print("------------------------")
    print("LEANING MODEL IS READY!!")
    costs = [] #cross-entropyの格納

# 学習の実行
# バッチはbatch_sizeだけランダムに学習セットから抽出して使用する。
    for ii in range(iternum):
        my_choice = np.random.choice(range(len(train_set_x)), batch_size)
        f1(train_set_x[my_choice], train_set_t[my_choice])
        costs.append(f_cost(train_set_x[my_choice], train_set_t[my_choice]))

        if ii%100 == 0:
            print("iter =",ii)
    print("COMPLETE!!🍻")

    return[w1,w2,b1,b2,costs]

def calc_accuracy(t,y):
#accuracyの計算:tと、yのargmaxの比較ベクトルを作り、そのsumが正解数になる。
    return np.sum( t == np.argmax(y,1) ) / len(t)

def testrun(params,datasets):
# 学習したパラメータを使って汎化性能のテストを行う。 
    w1 = params[0]
    w2 = params[1]
    b1 = params[2]
    b2 = params[3]
    costs = params[4]
    test_set_x, test_set_t = datasets

    x = T.dmatrix('x')
    h1 = T.nnet.relu( T.dot(x,w1) + b1 )
    h2 = T.nnet.relu( T.dot(h1,w2) + b2 )
    y = T.nnet.softmax( h2 )
    t = T.lmatrix('t')
    f2 = theano.function(inputs = [x], outputs = y)
    print("------------------------")
    print("TEST MODEL IS READY!!")

    accuracy = calc_accuracy( test_set_t , f2(test_set_x) )
    print("*********")
    print("OUR ACCURACY :", accuracy*100, " PER CENTO !!" )
    print("*********")
# loss-functionの描写
    plt.plot( costs ,'-')
    plt.show()

def main():
    datasets = load_data()
    params = init_params(datasets[2][0].shape[1], label_size(datasets[2][1]) )
    params = build_train(params, datasets[2])
    testrun(params,datasets[0])

if __name__ == "__main__":
    main()
