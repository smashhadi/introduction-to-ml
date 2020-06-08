"""
Keras based assignment
Modified code from https://github.com/keras-team/keras/tree/master/examples to answer questions

Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
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
model_name = 'keras_cifar10_model_cnn.h5'
model_name0 = 'keras_cifar10_model_0hidden.h5'
model_name1 = 'keras_cifar10_model_1hidden.h5'
model_name2 = 'keras_cifar10_model_2hidden.h5'
model_name3 = 'keras_cifar10_model_3hidden.h5'
model_name4 = 'keras_cifar10_model_4hidden.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#CNN
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
#512 rectified linear units
#0 hidden layers
model0 = Sequential()
model0.add(Flatten(input_shape=x_train.shape[1:]))
model0.add(Dense(num_classes))
model0.add(Activation('softmax'))
#1 hidden layer
model1 = Sequential()
model1.add(Flatten(input_shape=x_train.shape[1:]))
model1.add(Dense(512))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes))
model1.add(Activation('softmax'))
#2 hidden layers
model2 = Sequential()
model2.add(Flatten(input_shape=x_train.shape[1:]))
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))
#3 hidden layers
model3 = Sequential()
model3.add(Flatten(input_shape=x_train.shape[1:]))
model3.add(Dense(512))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))
model3.add(Dense(512))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))
model3.add(Dense(512))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))
model3.add(Dense(num_classes))
model3.add(Activation('softmax'))
#4 hidden layers
model4 = Sequential()
model4.add(Flatten(input_shape=x_train.shape[1:]))
model4.add(Dense(512))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
model4.add(Dense(512))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
model4.add(Dense(512))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
model4.add(Dense(512))
model4.add(Activation('relu'))
model4.add(Dropout(0.5))
model4.add(Dense(num_classes))
model4.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model0.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model4.compile(loss='categorical_crossentropy',
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
    model_info0 = model0.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)
    model_info1 = model1.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)
    model_info2 = model2.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)
    model_info3 = model3.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)
    model_info4 = model4.fit_generator(datagen.flow(x_train, y_train,
                                                  batch_size=batch_size),
                                     epochs=epochs,
                                     validation_data=(x_test, y_test),
                                     workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
model_path0 = os.path.join(save_dir, model_name0)
model0.save(model_path0)
model_path1 = os.path.join(save_dir, model_name1)
model1.save(model_path1)
model_path2 = os.path.join(save_dir, model_name2)
model2.save(model_path2)
model_path3 = os.path.join(save_dir, model_name3)
model3.save(model_path3)
model_path4 = os.path.join(save_dir, model_name4)
model4.save(model_path4)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
np.savetxt("cnn.csv", scores, delimiter=",")
scores0 = model0.evaluate(x_test, y_test, verbose=1)
np.savetxt("hidden0.csv", scores0, delimiter=",")
scores1 = model1.evaluate(x_test, y_test, verbose=1)
np.savetxt("hidden1.csv", scores1, delimiter=",")
scores2 = model2.evaluate(x_test, y_test, verbose=1)
np.savetxt("hidden2.csv", scores2, delimiter=",")
scores3 = model3.evaluate(x_test, y_test, verbose=1)
np.savetxt("hidden3.csv", scores3, delimiter=",")
scores4 = model4.evaluate(x_test, y_test, verbose=1)
np.savetxt("hidden4.csv", scores4, delimiter=",")

plt.figure()
#plt.plot(model_info.history['acc'],'r')
CS = plt.plot(model_info.history['val_acc'],'k')
CS = plt.plot(model_info0.history['val_acc'],'r')
CS = plt.plot(model_info1.history['val_acc'],'b')
CS = plt.plot(model_info2.history['val_acc'],'c')
CS = plt.plot(model_info3.history['val_acc'],'m')
CS = plt.plot(model_info4.history['val_acc'],'g')
CS = plt.xticks(np.arange(0, 11, 1.0))
#plt.rcParams['figure.figsize'] = (8, 6)
CS = plt.xlabel("Num of Epochs")
CS = plt.ylabel("Accuracy")
CS = plt.title("Test (Validation) Accuracy")
CS = plt.legend(['CNN', '0 hidden', '1 hidden', '2 hidden', '3 hidden', '4 hidden'], loc='upper left')
CS = plt.savefig('Q1_a.png')
