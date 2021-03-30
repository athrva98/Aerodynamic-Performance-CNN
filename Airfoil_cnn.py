

# Author : Athrva Atul Pandhare


import matplotlib.pyplot as plt
import scipy.io as dataloader
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.layers import BatchNormalization
import numpy as np
from keras.layers.core import Dropout

data = dataloader.loadmat(r"C:\Users\Athrva Pandhare\Desktop\Misc_Codes\Generative Adversarial Network\CNN-for-Airfoil-master\data\parsed_data\1_300.mat")
data_x, data_y, rNorm = data['data_x'], data['data_y'], data['Normalization_Factor']
num_data = np.shape(data_x)[0]
train_x, train_y = data_x[:int(0.7*num_data)], data_y[:int(0.7*num_data)]
valid_x,valid_y = data_x[int(0.2*num_data):int(0.9*num_data)], data_y[int(0.2*num_data):int(0.9*num_data)]
test_x, test_y = data_x[int(0.9*num_data):], data_y[int(0.9*num_data):]

class Neural_Net:
    def build(height, width, depth):
        chaDim = 1
        model = Sequential()
        model.add(Conv2D(32,(3,3), padding= "same", input_shape = (1,128,128)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chaDim))
        model.add(Conv2D(32,(3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chaDim))
        #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(64,(3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chaDim))
        model.add(Conv2D(64,(3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chaDim))
        model.add(Dropout(0.50))
        
        
        model.add(Flatten())
        model.add(Dense(70))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chaDim))
        model.add(Dropout(0.5))
        
        model.add(Dense(1))
        
        return model
def show_all_abs_error(trainX, model): # Takes a lot of time.
    abs_err = np.zeros(len(trainX))
    for i in range(len(trainX)):
        abs_err[i] = abs(trainY[i] - model.predict(trainX[i].reshape(1,1,128,128))[0])
    
    plt.scatter(range(len(trainX[-50:])), abs_err[-50:])
    
batch_size = 32
learning_rate = 0.0005
num_epochs = 30

opt2  = Adam(learning_rate = learning_rate, beta_1 = 0.9)
model = Neural_Net.build(height = 128,width = 128,depth = 1)
model.compile(loss="MeanSquaredError", optimizer=opt2)
trainX = []
testX = []
for X in train_x:
    trainX.append(X.reshape(1,128,128))
for Xt in test_x:
    testX.append(Xt.reshape(1,128,128))
trainX = np.array(trainX)
trainY = train_y
testX = np.array(testX)
testY = test_y
print("Training the Network...")
model.summary()
H = model.fit(trainX, trainY, validation_data= (testX,testY), batch_size = batch_size,
              epochs = num_epochs, verbose = 1)       

plt.figure()
plt.plot(np.arange(0, 35), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 35), H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()