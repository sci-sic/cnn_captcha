from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
import numpy as np
import glob
from PIL import Image
import random


def toonehot(strng, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector


###############################################
# Data loading
###############################################
random.seed(2018)

files = glob.glob('train/*.png')
files = [file for file in files if random.random()>=0.97]

train_data = np.stack([np.array(Image.open(file))/255.0 for file in files])
train_label = [file.split('_')[0] for file in files]
train_label = [lab.split('/')[1] for lab in train_label]
# print("Shape of train data:", train_data.shape)

read_label = [toonehot(label) for label in train_label]

train_label_final = [[] for i in range(5)]
for i in range(len(train_label)):
    train_label_final[0].append(read_label[i][0])
    train_label_final[1].append(read_label[i][1])
    train_label_final[2].append(read_label[i][2])
    train_label_final[3].append(read_label[i][3])
    train_label_final[4].append(read_label[i][4])

train_label = [arr for arr in np.asarray(train_label_final)]
print("Shape of train data:", train_data.shape)


files = glob.glob('test/*.png')
files = [file for file in files if random.random()>=0.98]

vali_data = np.stack([np.array(Image.open(file))/255.0 for file in files])
vali_label = [file.split('_')[0] for file in files]
vali_label = [lab.split('/')[1] for lab in vali_label]
print("Shape of vali data:", vali_data.shape)

read_label = [toonehot(label) for label in vali_label]

vali_label_final = [[] for i in range(5)]
for i in range(len(vali_label)):
    vali_label_final[0].append(read_label[i][0])
    vali_label_final[1].append(read_label[i][1])
    vali_label_final[2].append(read_label[i][2])
    vali_label_final[3].append(read_label[i][3])
    vali_label_final[4].append(read_label[i][4])

vali_label = [arr for arr in np.asarray(vali_label_final)]

###############################################
# Building model
###############################################

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(150, 330, 3))  # this assumes K.image_data_format() == 'channels_last'

# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 36 classes
# predictions = Dense(36, activation='softmax')(x)

predictions = [Dense(36, name='digit1', activation='sigmoid')(x),\
    Dense(36, name='digit2', activation='sigmoid')(x),\
    Dense(36, name='digit3', activation='sigmoid')(x),\
    Dense(36, name='digit4', activation='sigmoid')(x),\
    Dense(36, name='digit5', activation='sigmoid')(x)]


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
filepath = "cnn_inception.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit6_acc', verbose=1, save_best_only=True, mode='max',period=5)
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
callbacks_list = [tensorBoard , checkpoint]

model.fit(train_data, train_label, batch_size=50, epochs=4, verbose=2, 
    validation_data=(vali_data, vali_label), callbacks=callbacks_list)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False


for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit(train_data, train_label, batch_size=60, epochs=100, verbose=2, 
    validation_data=(vali_data, vali_label), callbacks=callbacks_list)

model.save_weights("cnn_inception.h5")

