# Script for Kaggel Digit Recognizer Competition
# Author: Manuel Spierenburg


import keras
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


print("1. loading data")
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

Y_train = train.values[:,0]
X_train = train.values[:,1:]
X_test = test.values

y_train = to_categorical(Y_train)
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(test.shape[0], 28, 28, 1)



print("2. running cnn")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])
model.fit(x_train, y_train,
        batch_size=128,
        validation_split=0.1)

pred = model.predict(x_test)

# select index with max prob
pred = np.argmax(pred,axis = 1)
sub = pd.DataFrame({"ImageId": range(1,len(pred)+1), "Label": pred})
sub.to_csv("submission_cnn.csv", index=False)