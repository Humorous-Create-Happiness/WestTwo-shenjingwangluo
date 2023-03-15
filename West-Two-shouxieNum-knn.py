from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#训练集
readCsv1= pd.read_csv(r'C:\Users\Lenovo\Desktop\py\num\train.csv', sep=',')
train=np.array(readCsv1)

train_label=[]
train_target=[]
target=[]

for i in train:
    train_label.append(i[0])
    target=i[1:]
    train_target.append(target)
train_label=np.array(train_label)
train_target=np.array(train_target)

#测试集
readCsv2= pd.read_csv(r'C:\Users\Lenovo\Desktop\py\num\test.csv', sep=',')
test=np.array(readCsv2)

readCsv3= pd.read_csv(r'C:\Users\Lenovo\Desktop\py\num\sample_submission.csv', sep=',')

test_target=[]

for i in test:
    target=i[0:]
    test_target.append(target)
test_target=np.array(test_target)
test_label=np.array(readCsv3)
print(type(test_target))
#构建并训练模型

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_target, train_label)
y_pred = knn.predict(test_target)

# 评估模型
print("模型精度：{:.2f}".format(np.mean(y_pred == test_label)))
print("模型精度：{:.2f}".format(knn.score(test_target, test_label)))