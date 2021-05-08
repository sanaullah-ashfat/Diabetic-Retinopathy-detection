# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 00:13:21 2019

@author: Sanaullah Chowdhury
"""
import os
import fnmatch 
import cv2 
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.manifold import TSNE
from keras import backend as K
from keras import objectives
from keras.layers import Conv2D,concatenate, Dropout,MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as plot
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator



train_csv = pd.read_csv("G:/blind_detection/train.csv")
test_csv = pd.read_csv("G:/blind_detection/test.csv")
train_path = "G:/blind_detection/example"
#xxx = pd.DataFrame(train_csv)
#
#
#x =[298,2374,556,3450,1868,2026,2182,3448,676,2660,917,1152,1994,169,1033,1131,
#    2210,3547,985,3272,853,2980,306,1089,2691,3220,915,1381,1175,3622,3340,3470,
#    2669,3265,713,1074,392,3360,844,1643,682,2826,1374,1679,3606,34,815,854,864,1589,658,2697,
#    384,745,2159,3320,274,3462,663,2414,1355,2431,2236,2555,1351,2297,1810,2455,146,3615,1896,2766,3100,
#    1397,1672,2954,2982]

df = train_csv.drop(train_csv.index[[85,344,493,584,791,947,1207,1223,1281,1617,1946,2082,2098,2113]])
df =df.reset_index(drop=True)

df["id_code"] = df["id_code"].apply(lambda x :os.path.join(train_path,"{}.png".format(x)))
#train_csv["diagnosis"] = train_csv["diagnosis"].astype("str")


img_height,img_width =128,128

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]
    
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    
    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

   
    elif zoom_factor > 1:

        
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

   
    else:
        out = img

def get_train_image(image,augmentation =False):

    train =np.array(image) 
      
    if augmentation:
        all_train_img = [train]
        flip_train = train[:,::-1,:]
        all_train_img.append(flip_train)
        
        
        for i in range(0,360,90):
            all_train_img.append(rotate(train,i,axes=(1, 2), reshape=False))
            
            all_train_img.append(rotate(flip_train,i,axes=(1, 2), reshape=False))
            
        train = np.concatenate(all_train_img,axis =0)    
             
        #z score and mean std
        
        all_images = train.shape[0]
        for j in range(all_images):
            mean = np.mean(train[j,...][train[j,...,0]>40.0],axis =0)
            print(mean)
            std = np.std(train[j,...][train[j,...,0]>40.0],axis =0)
            print(std)
            assert len(mean)==3 and len(std)==3
            train[j,...]=(train[j,...]-mean)/std
#            
        return train

y=df['diagnosis']

img_list=[]
#list1=[]
#list2=[]
#list3=[]
#list4=[]
#list5=[]
#
x = len(df["id_code"])
i=0
for i in range(x):
    path =df["id_code"][i]
    img = cv2.imread(path)

#    
    img_list.append(img)

X = get_aug_image(img_list,augmentation =True)

yy =y.repeat(6)

Y = np.concatenate([y,y,yy,yy])
#image = img_list+list1+list2+list3+list4+list5

#image = img_list+list_360
#y =np.concatenate([Y,Y,Y,Y,Y,Y])
#y =np.concatenate([Y,Y])
#del img_list,list1,list2,list3,list4,list5
x_train,x_test,y_train,y_test = train_test_split(img_list,Y,test_size=.15,stratify =Y, random_state=8)

X_train = np.array(x_train)
X_test = np.array(x_test)
#
##X_train_z = clipped_zoom(X_train,1.03)
###
#X_train_lr =np.fliplr(x_train)
####
###X_train_f =np.flip(x_train,1) 
####
#X_train=np.concatenate([X_train,X_train_lr])
####
####
###del X_train_lr
###del X_train_f
####
#####X_test_z = clipped_zoom(X_test,1.15)
####
#X_test_lr =np.fliplr(x_test)
####
###X_test_f =np.flip(x_test,1) 
####
#X_test=np.concatenate([X_test,X_test_lr])
####
##del X_test_lr
##del X_train_lr
###
###del img
##
####
#y_train =np.concatenate([y_train,y_train])
#y_test =np.concatenate([y_test,y_test])
###



X_train = X_train.reshape(X_train.shape[0],img_height,img_width,3)
X_test = X_test.reshape(X_test.shape[0],img_height,img_width,3)



X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
#print(X_test)
X_mean, X_std = np.mean(X_train), np.std(X_train)
X_train, X_test = (X_train - X_mean) / X_std, (X_test - X_mean) / X_std




print('X_train shape', X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')


Y_train=np_utils.to_categorical(y_train, 5)
Y_test=np_utils.to_categorical(y_test,5)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#reader_label=pd.read_csv("data_tamurah_normalized_shuffle.csv", header=None)

#from matices_1 import precision,recall


def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
    
def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
    
def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def matthews_correlation(y_true, y_pred):
    """Matthews correlation metric.
    It is only computed as a batch-wise average, not globally.
    Computes the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1))






#img_height,img_width =128,128

#img_width= img_rows
#img_height = img_cols
classes_num = 5
epochs =25
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
lr = 0.0004


model = Sequential()
model.add(Conv2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_height,img_width,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dropout(0.8))

model.add(Dense(classes_num, activation='softmax'))

#model.compile(optimizer='adam', loss="categorical_crossentropy",  metrics=["accuracy"])

#STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
#
#history_finetunning = model.fit_generator(generator=train_generator,
#                              steps_per_epoch=STEP_SIZE_TRAIN,
#                              validation_data=valid_generator,
#                              validation_steps=STEP_SIZE_VALID,
#                              epochs=EPOCHS,
#                              verbose=1).history



model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) 

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=64,validation_data=(X_test, Y_test))
    
    





from sklearn.metrics import confusion_matrix

Y_prediction = model.predict(X_test)
print(len(Y_prediction))
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(30,25))
plt.title('Confusion Matrix ')
ax =plt.subplot()
sns.heatmap(confusion_mtx, annot=True, fmt = 'd',cmap ='BuPu')

plt.show()



from efficientnet import EfficientNetB3
from keras.activations import elu
from keras.optimizers import Adam

effnet = EfficientNetB3(weights=None,
                        include_top=False,
                        input_shape=(img_height,img_width, 3))
#effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')

def build_model():
    """
    A custom implementation of EfficientNetB3
    for the APTOS 2019 competition
    (Regression)
    """
    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation=elu))
    model.add(Dense(5, activation="linear"))
    model.compile(loss='mse',
                  optimizer=Adam(0.0001), 
                  metrics=['mse', 'acc'])
    print(model.summary())
    return model

# Initialize model
model = build_model()



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=64,validation_data=(X_test, Y_test))
    








classes_num = 5
epochs =50
xc=epochs
model =Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(380,name ="dense_121"))
model.add(Activation('relu'))
#model.add(Dense(100))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(classes_num ))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
print('model complied!!')



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=64,validation_data=(X_test, Y_test))
    


input_img = Input((img_height,img_width, 3))

conv1 = Conv2D(filters =8,kernel_size=(3,3),activation ="relu")(input_img)
#a = conv1._keras_shape
conv2 = Conv2D(filters =16,kernel_size=(3,3),activation ="relu")(conv1)
#b = conv2._keras_shape
conv3 = Conv2D(filters =32,kernel_size=(3,3),activation ="relu")(conv2)
c = conv3._keras_shape
conv4 = Reshape((int(c[1]/2),int(c[2]/2),128))(conv3)
#x = conv4._keras_shape
conv5 = Conv2D(filters =64,kernel_size = (3,3),activation = "relu")(conv4)
#d = conv5._keras_shape


flatten_layer = Flatten()(conv5)

dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.1)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.1)(dense_layer2)
output_layer = Dense(classes_num=5, activation='softmax')(dense_layer2)

model2 = Model(inputs=input_img, outputs=output_layer)




adam = Adam(lr=0.001, decay=1e-06)
model2.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



datagen = ImageDataGenerator(
        rotation_range=0.5, 
        zoom_range = 0.5, 
        width_shift_range=0.5,  
        height_shift_range=0.5, 
        horizontal_flip=True, 
        vertical_flip=True)

datagen.fit(X_train)

history=model2.fit_generator(datagen.flow(X_train,y_train, batch_size=200),
                              epochs = 20, validation_data = (X_test,y_test), steps_per_epoch=500)



#
#

test_path ="G:/blind_detection/extest"

#
#dff = train_csv.drop(train_csv.index[[85,344,493,584,791,947,1207,1223,1281,1617,1946,2082,2098,2113]])
#df =df.reset_index(drop=True)

test_csv["id_code"] = test_csv["id_code"].apply(lambda x :os.path.join(test_path,"{}.png".format(x)))
#train_csv["diagnosis"] = train_csv["diagnosis"].astype("str")



img_height,img_width =128,128


test_list =[]

xx = len(df["id_code"])
i=0
for i in range(xx):
    path =df["id_code"][i]
    img = cv2.imread(path)
    
    test_list.append(img)


image = np.array(test_list)

xt = model.predict(image)


dg =np.argmax(xt,axis=1)

dd = dg[0:1928]

len(dd)

bb = test_csv["id_code"].copy()


cc =pd.DataFrame({"id_code": bb, "diagnosis": dd})

cc.to_csv("G:/blind_detection/submission1.csv")









