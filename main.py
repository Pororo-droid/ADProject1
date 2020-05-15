import tensorflow as tf
import pandas as pd
import os

######### 파일 목록 불러오기
path = "./"
file_list = os.listdir(path)
file_list_xlsx = [file for file in file_list if file.endswith(".xlsx")]

print("목록: ")
print(file_list_xlsx)
filename = input("읽으실려는 파일 이름을 입력해주세요 : ")

data = pd.read_excel(filename)


########## 파일 데이터 불러오기

dataX = []
dataY = []
index = 1

for i in data['누적관객수']:
    dataX.append(index)
    dataY.append(i)
    index+=1

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX = dataX[0:train_size]
trainY = dataY[0:train_size]

testX = dataX[train_size:len(dataX)]
testY = dataY[train_size:len(dataY)]

########### 인공지능 학습 시키기
W = tf.Variable(tf.random_normal(shape = [1]),name = "W")
b = tf.Variable(tf.random_normal(shape = [1]),name = "b")
x = tf.placeholder(tf.float32,name = "x")

### 범위
max_diff_range = (testY[-1])//10
epoch = 500000

###
linear_model = W*x+b

y = tf.placeholder(tf.float32,name = "y")

loss = tf.reduce_mean(tf.square(linear_model-y))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(loss)




"""
while True:
    for i in range(epoch):
        sess.run(train_step, feed_dict={x: trainX, y: trainY})

    for i in range(test_size):
        if abs(sess.run(linear_model,feed_dict={x:testX[i]}) - testY[i]) > max_diff_range:
            print(i)
            break
    else:
        break
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(epoch):
    sess.run(train_step,feed_dict={x:trainX, y:trainY})

while True:
    for i in range(test_size):
        if abs(sess.run(linear_model, feed_dict={x: testX[i]}) - testY[i]) > max_diff_range:
            #print(sess.run(linear_model, feed_dict={x: testX[i]}), testY[i] , abs(sess.run(linear_model, feed_dict={x: testX[i]}) - testY[i]))
            trainX.append(testX.pop(0))
            trainY.append(testY.pop(0))
            test_size = len(testY)
            break
    else:
        break
    for step in range(epoch):
        sess.run(train_step, feed_dict={x: trainX, y: trainY})

########## 결과값 내기
x_test = [index]

print("지금으로부터 1일 후의 예상 누적관객수는 %d 입니다"%(sess.run(linear_model,feed_dict={x:x_test})))

sess.close()