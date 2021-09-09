import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#%matplotlib inline

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel("Acc")
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show
    

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap = 'binary')
        title = "label = " + str(labels[idx])
        if len(prediction)>0:
            title += ", predict = " +str(prediction[idx]) 
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()
    
#----------------------------------------------------------------------------------------------


(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)


model = Sequential()

model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1000,
                kernel_initializer='normal',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(x=x_Train_normalize,
                          y=y_Train_OneHot, 
                          validation_split=0.25,
                          epochs=10,
                          batch_size=200,
                          verbose=2)






show_train_history(train_history, 'accuracy', 'val_accuracy')

scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print()
print()
print('accuracy = ', scores[1])
print('loss = ', scores[0])


#-----------------------------------------------------------------------
#prediction = model.predict_classes(x_Test)
#print(prediction)
#plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340, num=25)

#pd.crosstab(y_test_label, prediction, colnames=['predict'], rownames=['label'])