import pandas as pd
import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#データの読み込み
train = pd.read_csv('./train.csv')
train_d = train.dropna()
test = pd.read_csv('./test.csv')


#使用する特徴量の選択
t_col_name = ['Survived']
x_col_name = ['Pclass','Sex', 'SibSp','Parch','Embarked']

#質的特徴の変換
t_train = np.array(train_d['Survived'])
t_test = np.array(train_d['Survived'])

tmp = train_d[x_col_name].replace('male',1).replace('female',2)
x_train = np.array(tmp[x_col_name].replace('S',1).replace('C',2).replace('Q',3))
tmp = test[x_col_name].replace('male',1).replace('female',2) 
x_test = np.array(tmp[x_col_name].replace('S',1).replace('C',2).replace('Q',3)) 

#特徴量の標準化
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#SVM
model = SVC(kernel='linear', random_state=None)
model.fit(x_train_std, t_train)

pred_test = model.predict(x_test_std)

tmp = np.array([np.array(test['PassengerId']),pred_test]).T
df = pd.DataFrame(data = tmp, columns=['PassengerId','Survived'])
df.to_csv('./result.csv',index=False)