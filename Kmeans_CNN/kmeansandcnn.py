# import 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scikitplot as skplt
import keras

import csv
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from itertools import cycle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from keras import backend
from numpy import array

from keras import backend as K



def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
	
def top5acc(y_true,y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)

def create_model():
	# create model
    model = Sequential()
    model.add(Convolution2D(num_pixels, 3, 3, input_shape=input_shape, activation= "relu"))
       
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(num_classes, activation= "softmax"))
	# compile modelr=0.01,momentum=0.9
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy',rmse,top5acc])
    return model

'''
sam = pd.read_csv('Analysis.csv')

mytrain=open('cnntrainlabel.csv','a')
wr=csv.writer(mytrain,quoting=csv.QUOTE_ALL)

mytest=open('cnntestlabel.csv','a')
wrt=csv.writer(mytest,quoting=csv.QUOTE_ALL)


le = preprocessing.LabelEncoder()

for col in sam.columns.values:
	if sam[col].dtypes=='object':
		le.fit(sam[col])
		sam[col]=le.transform(sam[col])



#data = [sam1,sam2,sam3,sam4]
#data = pd.concat(sam)
pre_data = sam
#print(data)
kmeans = KMeans(n_clusters=3, random_state=0).fit(pre_data)
class_labels = kmeans.labels_

pre_data = pre_data.values.tolist()
#class_labels = class_labels.values.tolist()
	 
print("data", len(pre_data))
print("labels",len(class_labels))
fg = 0
thres = int(len(pre_data)*0.7)

for i in range(0,len(pre_data)):
    pre_data[i].append(0)
    pre_data[i].append(0)
    pre_data[i].append(0)
    pre_data[i].append(0)
    pre_data[i].append(0)
    pre_data[i].append(class_labels[i])
    if fg < thres:
        wr.writerow(pre_data[i])
        fg = fg + 1
    else:
        wrt.writerow(pre_data[i])
        fg = fg + 1
'''
seed = 9
np.random.seed(seed)

train = pd.read_csv('cnntrainlabel.csv')
test  = pd.read_csv('cnntestlabel.csv')

images = train.iloc[:,:16].values
scaler = StandardScaler().fit(images) 
rescaledtrain = scaler.transform(images) 

labels = train.iloc[:,16].values

test_images = test.iloc[:,:16].values
scalr = StandardScaler().fit(test_images) 
rescaledtest = scalr.transform(test_images) 

imags = pd.DataFrame(data = rescaledtrain)
test_imags = pd.DataFrame(data = rescaledtest)

img_rows,img_cols=4,4
x_train=imags.as_matrix()
x_test=test_imags.as_matrix()

if K.image_data_format() == 'channels_first': 
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,4,4)    
else:
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape=(4,4,1) 

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

# encode the training labels as integers
encoded_labels = np_utils.to_categorical(labels)
num_classes = encoded_labels.shape[1]
y_train=labels

y_test=test.iloc[:,16].values
ye_test = y_test

num_pixels=4
num_classes=3
	
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
	

model = create_model()
history = model.fit(x_train,y_train,validation_split=0.2,shuffle=True,epochs = 100,batch_size = 10,verbose=2)
		
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(1)

mytrain=open('cnnresults.csv','a')
results =csv.writer(mytrain,quoting=csv.QUOTE_ALL)

rmseval = history.history['rmse']
top5acc = history.history['top5acc']
trainacc = history.history['acc']
valacc = history.history['val_acc']
trainloss = history.history['loss']
valloss = history.history['val_loss']


for i in range(0,100):
    res = [i,trainacc[i],valacc[i],trainloss[i],valloss[i],rmseval[i],top5acc[i]]
    results.writerow(res)
    

plt.title('RMSE vs Epochs')
plt.plot(history.history['rmse'])
plt.ylabel('RMSE')
plt.xlabel('Epochs')
plt.show()

plt.title('Top-5 Accuracy vs Epochs')
plt.plot(history.history['top5acc'])
plt.ylabel('Top-5 Accuracy')
plt.xlabel('Epochs')
plt.show()
#precision = precision_score(ye_test, y_pred, average=None)
#recall = recall_score(ye_test, y_pred, average=None)
#accuracy = accuracy_score(ye_test, y_pred)

#y_score = model.evaluate(x_test, y_test,batch_size=10)
#print('y_score ',y_score)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

'''
##############
graph_cl = [0,1,2]
plt.plot(graph_cl,precision,color='g')
plt.xlabel('Classes')
plt.ylabel('Precision')
plt.show()

#plt.scatter(graph_cl,recall,c=graph_cl)
plt.plot(graph_cl,recall,c='orange')
plt.xlabel('Classes')
plt.ylabel('Recall')
plt.show()

print("y_test ",y_test)
print("ye_test ",ye_test)
print("y_pred ",y_pred)
#############
'''
