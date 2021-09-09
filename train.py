# Train trên colab
# from google.colab import drive
# drive.mount('/content/gdrive')

from os import listdir
import cv2 as cv
# from google.colab.patches import cv2_imshow
import os
pixels = []
labels = []
# Tạo dataset
for forder in listdir('/content/gdrive/MyDrive/NhanDienTien'):
  path = '/content/gdrive/MyDrive/NhanDienTien/' + forder
  for file in listdir(path):
    pathfile = path + '/' + file
    print('File: ', file)
    pixels.append(cv.resize(cv.imread(pathfile), dsize=(128, 128)))
    labels.append(forder)
# Lưu dataset
import pickle
from sklearn.preprocessing import LabelBinarizer
import numpy as np

pixels = np.array(pixels)
labels = np.array(labels)

coder = LabelBinarizer()
labels = coder.fit_transform(labels)

file = open('/content/gdrive/MyDrive/NhanDienTien/pix.data', 'wb')
pickle.dump((pixels, labels), file)

file.close()

# Mở dataset
with open('/content/gdrive/MyDrive/NhanDienTien/pix.data', 'rb') as pickle_file:
    pixels, labels = pickle.load(pickle_file)

# Xây dựng model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Dong bang cac layers o feature extraction
for layer in model_vgg16_conv.layers:
  layer.trainable = False

input = Input(shape=(128, 128, 3))
output_vgg16_conv = model_vgg16_conv(input)

# them cac FC va dropout de tranh overfit
x = Flatten()(output_vgg16_conv)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4, activation='softmax')(x)

model = Model(inputs=input, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Chia dữ liệu train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.2, random_state=42)
X_test.shape

# Lưu lại weight
from keras.callbacks import ModelCheckpoint
filepath = '/content/gdrive/MyDrive/NhanDienTien/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkPoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback_lists = [checkPoint]

# Train dữ liệu trên colab
# augement image for train model
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1, rescale=1. /255, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         brightness_range=[0.25, 1.5], fill_mode='nearest')
model.fit_generator(aug.flow(X_train, y_train, batch_size=64), epochs=50, validation_data=aug.flow(X_test, y_test, batch_size=64), callbacks=callback_lists)
model.save('/content/gdrive/MyDrive/NhanDienTien/my_model.h5')

