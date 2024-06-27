#@title Run this to download the data and setup our environment
import cv2
import dlib
import gdown
import pickle
import warnings
import itertools

import numpy as np
import pandas as pd
import seaborn as sns

import urllib.request
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import re
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Conv3D
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# grab tools from our tensorflow and keras toolboxes!
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

#Integer to Label Mapping
label_map = {0:"ANGRY",1:"HAPPY",2:"SAD",3:"SURPRISE",4:"NEUTRAL"}

#Load the datala
df = pd.read_csv("./fer2013_5.csv")
print(df.head())

# generate x labels for our plot
emotion_labels = [label_map[i] for i in label_map.keys()]

# generate counts for each emotion type
emotion_counts = [np.sum(df["emotion"] == i) for i in range(len(label_map))]

# generate a bar plot for our emotion labels that has different colors
# [plt.bar(x = emotion_labels[i], height = emotion_counts[i] ) for i in range(len(emotion_labels))]

# make the plot interpretable with x and y labels + title
# plt.xlabel('EMOTION LABEL')
# plt.ylabel('N OBSERVSATIONS')
# plt.title('A balanced distribution of emotions in our data set', y=1.05);

# the number of times we pass all the training data through the model
epochs = 10
# the number of examples we pass to the model at each time
batch_size = 64
# the proportion of testing data we set aside (e.g. 10%)
test_ratio = .1
# the number of emotion categories we have to predict
n_labels = 5

# set to True if we want to preload data -- which has already been generated for us :)
preload = True

if preload:

  # load outputs saved in this folder after running preprocess_data()
  dataX = np.load('./pureX.npy')
  dataY = np.load('./dataY.npy', allow_pickle=True)

else:
    pass
  # this takes 15-20 minutes to run, but someone has already run it and saved the ouputs in this folder
  # pureX, dataX, dataY = preprocess_data(df)


y_onehot = to_categorical(dataY, len(set(dataY)))

#Split Data into Train, Test (90-10)
X_train, X_test, y_train, y_test = train_test_split(dataX, y_onehot, test_size=0.1, random_state=42,)
'''
####Standardize the data####################
###Note: Do not use test data to fit your Standardscaler Model
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
'''
#Reduces features by maintaining 95% variance of the data
#After doing PCA on our training data, 2278 Dimensions --->reduced to 20
#Note: PCA is trained only on training data
pca = PCA(.95)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test= pca.transform(X_test)
'''

# we'll use the same epochs and batch size as above
width, height = 48, 48


X_train_cnn = X_train.reshape(len(X_train),height,width)
X_test_cnn = X_test.reshape(len(X_test),height,width)

X_train_cnn = np.expand_dims(X_train_cnn,3)
X_test_cnn = np.expand_dims(X_test_cnn,3)

X_train_cnn = np.concatenate((X_train_cnn,X_train_cnn,X_train_cnn),axis=3)
X_test_cnn = np.concatenate((X_test_cnn,X_test_cnn,X_test_cnn),axis=3)

print(X_train_cnn.shape,X_test_cnn.shape,"IDJOAEIJWDIPJWAPKODOKPWAD")
class Model:
    def __init__(self, name,n_labels:int = 5):
        self.model = Sequential()
        self.path = name+".h5"
        self.checkpoint = ModelCheckpoint(self.path, verbose=1, monitor='val_loss', save_best_only=True,
                                     mode='auto')
        self.n_labels = n_labels

    def initalize_model(self,width,height):
        print("BAD")
        pass

    def train_model(self):
        # training the model
        self.history = self.model.fit(X_train_cnn, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                    callbacks=[self.checkpoint], validation_data=(X_test_cnn, y_test), shuffle=True)

    def evaluate(self):
        performance = self.model.evaluate(X_test_cnn, y_test)
        print(performance)
        self.model.save(self.path)
        return performance

class VGG(Model):
    def initalize_model(self,width,height):
        model = tf.keras.applications.VGG19(input_shape=(width,height,3),include_top=False,weights="imagenet")
        for layer in model.layers[:-4]:
            layer.trainable=False
        self.model.add(model)
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(32,kernel_initializer="he_uniform"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32,kernel_initializer="he_uniform"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, kernel_initializer="he_uniform"))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dense(5,activation="softmax"))
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                          metrics=[
                              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                              tf.keras.metrics.Precision(name='precision'),
                              tf.keras.metrics.Recall(name='recall'),
                              tf.keras.metrics.AUC(name='auc')
                          ])

class Mayur(Model):
    def initalize_model(self,width,height):
        # this conv layer has 64 filters! the input shape needs to be the same dimensions of the image
        self.model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=(width, height, 1)))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        self.model.add(Conv2D(64, kernel_size=(7, 7), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
        self.model.add(Conv2D(128, kernel_size=(7, 7), activation='relu'))
        # self.model.add(Conv2D((32,32), kernel_size=(3, 3), activation='relu'))

        # batch normalization
        # self.model.add(BatchNormalization())
        # max pooling
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        # dropout
        self.model.add(Dropout(0.25))

        # flatten all the outputs between convolutional and dense layers
        self.model.add(Flatten())
        # add a "dense layer" (i.e. the fully connected layers in MLPs) with dropout
        self.model.add(Dense(296, activation='relu'))

        self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dense(128, activation='sigmoid'))
        # output layer
        self.model.add(Dense(self.n_labels, activation='softmax'))
        # Saves the Best Model Based on Val Loss

        # compliling the model with adam optimizer and categorical crossentropy loss
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                          metrics=['accuracy'])


model = VGG("best-vgg-19-model")
model.initalize_model(width=width,height=height)
model.train_model()
model.evaluate()

