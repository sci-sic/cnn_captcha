
import os
import glob
import csv
import random
import numpy as np
from PIL import Image
from numpy import argmax
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


def toonehot(strng, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector



# inverted = [int_to_char[argmax(i)] for i in x]
# print(inverted)


# Create CNN Model
print("Creating CNN model...")
tensor_in = Input((150, 330, 3))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = [Dense(36, name='digit1', activation='softmax')(tensor_out),\
    Dense(36, name='digit2', activation='softmax')(tensor_out),\
    Dense(36, name='digit3', activation='softmax')(tensor_out),\
    Dense(36, name='digit4', activation='softmax')(tensor_out),\
    Dense(36, name='digit5', activation='softmax')(tensor_out)]
model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()

# print("Reading training data...")
# train_data = np.stack([np.array(Image.open("./data/train_set/" + str(index) + ".jpg"))/255.0 for index in range(1, 50001, 1)])
# traincsv = open('./data/train_set/train.csv', 'r', encoding = 'utf8')
# read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
# train_label = [[] for _ in range(6)]
# for arr in read_label:
#     for index in range(6):
#         train_label[index].append(arr[index])
# train_label = [arr for arr in np.asarray(train_label)]
# print("Shape of train data:", train_data.shape)

###############################################
# Data loading
###############################################

files = glob.glob('train/*.png')
files = [file for file in files if random.random()>=0.98]

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


filepath="cnn_model_2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit5_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
callbacks_list = [tensorBoard, earlystop, checkpoint]
model.fit(train_data, train_label, batch_size=100, epochs=200, verbose=2, 
	validation_data=(vali_data, vali_label), callbacks=callbacks_list)
model.save_weights("cnn_model.h5")

# # serialize model to JSON
# model_json = model.to_json()
# with open("cnn_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("cnn_model.h5")
# print("Saved model to disk")

