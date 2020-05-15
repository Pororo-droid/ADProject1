import tensorflow as tf
import pandas as pd
import os
import numpy as np

######### 파일 목록 불러오기
path = "./"
file_list = os.listdir(path)
file_list_xlsx = [file for file in file_list if file.endswith(".xlsx")]

print("목록: ")
print(file_list_xlsx)

#filename = input("읽으실려는 파일 이름을 입력해주세요 : ")
filename = "KOBIS_에이지_오브_울트론.xlsx"
data = pd.read_excel(filename)

n= 10
#n = int(input("지금으로부터 몇일 후의 누적관객수를 보시겠습니까? : "))
########## 파일 데이터 불러오기

dataX = []
dataY = []
index = 1

for i in data['누적관객수']:
    dataX.append(i)
    dataY.append(index)
    index+=1

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
z
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

########### 인공지능 학습 시키기
W = tf.Variable(tf.random_normal(shape = [1]),name = "W")
b = tf.Variable(tf.random_normal(shape = [1]),name = "b")
x = tf.placeholder(tf.float32,name = "x")
max_diff_range = 100000

linear_model = W*x+b

y = tf.placeholder(tf.float32,name = "y")

loss = tf.reduce_mean(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(0.001)

train_step = optimizer.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train_step, feed_dict={x: trainX, y: trainY})


########## 결과값 내기
x_test = [index+n]

print("지금으로부터 %d일 후의 예상 누적관객수는 %f 입니다"%(n,sess.run(linear_model,feed_dict={x:x_test})))

sess.close()