import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

df = pd.read_csv('train.csv')
data = np.array(df.values)

pics = data[:,1:]
m, n = pics.shape
pics = pics.reshape(m, 28, 28, 1)
pics = pics / 255

labels = np_utils.to_categorical(data[:,0])

batch_size = 16

# Build the simple CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(pics, labels, batch_size=batch_size, epochs=20, verbose=1)

model.save_weights('my_weights.h5')



tdf = pd.read_csv('test.csv')
tpics = np.array(tdf.values)

tm, tn = tpics.shape
tpics = tpics.reshape(tm, 28, 28, 1)
tpics = tpics / 255

my_pred = model.predict_classes(tpics, batch_size=batch_size)

wdf = pd.DataFrame(columns=('ImageId' , 'Label'))
for i in range(tm):
   wdf.loc[i] = [i+1, my_pred[i]] 

wdf.to_csv('submission.csv', index=False)
