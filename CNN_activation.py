"""
Keras based assignment
Compare performance RLU and sigmoid as activation layers in CNN
Code following https://github.com/keras-team/keras/tree/master/examples.
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
num_classes = 10
epochs = 10
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_model_cnn_relu.h5'
model_name_s = 'keras_cifar10_model_cnn_sigmoid.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#With Sigmoid
model_s = Sequential()
model_s.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model_s.add(Activation('sigmoid'))
model_s.add(Conv2D(32, (3, 3)))
model_s.add(Activation('sigmoid'))
model_s.add(MaxPooling2D(pool_size=(2, 2)))
model_s.add(Dropout(0.25))

model_s.add(Conv2D(64, (3, 3), padding='same'))
model_s.add(Activation('sigmoid'))
model_s.add(Conv2D(64, (3, 3)))
model_s.add(Activation('sigmoid'))
model_s.add(MaxPooling2D(pool_size=(2, 2)))
model_s.add(Dropout(0.25))

model_s.add(Flatten())
model_s.add(Dense(512))
model_s.add(Activation('sigmoid'))
model_s.add(Dropout(0.5))
model_s.add(Dense(num_classes))
model_s.add(Activation('softmax'))

#With Rectified Linear Units
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model_s.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model_info = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model_info = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
    model_info_s = model_s.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
#print('Saved trained model at %s ' % model_path)
model_path_s = os.path.join(save_dir, model_name_s)
model_s.save(model_path_s)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
np.savetxt("cnn_relu.csv", scores, delimiter=",")
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
scores_s = model_s.evaluate(x_test, y_test, verbose=1)
np.savetxt("cnn_sigmoid.csv", scores_s, delimiter=",")

plt.figure(0)
#plt.plot(model_info.history['acc'],'r')
CS = plt.plot(model_info.history['val_acc'],'g')
CS = plt.plot(model_info_s.history['val_acc'],'r')
CS = plt.xticks(np.arange(1, 10, 1.0))
#plt.rcParams['figure.figsize'] = (8, 6)
CS = plt.xlabel("Num of Epochs")
CS = plt.ylabel("Accuracy")
CS = plt.title("Test (Validation) Accuracy")
CS = plt.legend(['CNN with relu','CNN with sigmoid'], loc='upper left')
plt.savefig('Q1_b.png')
