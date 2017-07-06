import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications

df = pd.read_csv('test.csv')
pics = np.array(df.values)

m, n = pics.shape
pics = pics.reshape(m, 28, 28, 1)
pics = pics / 255

batch_size = 100

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


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.load_weights('my_weights.h5')

my_pred = model.predict_classes(pics, batch_size=batch_size)

wdf = pd.DataFrame(columns=('ImageId' , 'Label'))
for i in range(m):
   wdf.loc[i] = [i+1, my_pred[i]] 

wdf.to_csv('submission.csv', index=False)
