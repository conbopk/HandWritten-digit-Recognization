import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL.ImageOps import grayscale
from keras import Sequential
from keras.src.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
import cv2
import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam

# get data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val, y_val = x_train[50000:60000,:], y_train[50000:60000]
x_train, y_train = x_train[:50000,:], y_train[:50000]

# new data from my dir
train_dir = 'Datasets/handwritten_digit/train'
val_dir = 'Datasets/handwritten_digit/val'

# new label y for train and valid set
train_folders = os.listdir(train_dir)
val_folders = os.listdir(val_dir)

train_labels = []
val_labels = []

for folder in train_folders:
    folder_path = os.path.join(train_dir,folder)
    images = os.listdir(folder_path)
    for image in images:
        train_labels.append(int(folder))

for folder in val_folders:
    folder_path = os.path.join(val_dir,folder)
    images = os.listdir(folder_path)
    for image in images:
        val_labels.append(int(folder))

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

y_train_new = keras.utils.to_categorical(train_labels, 10)
y_val_new = keras.utils.to_categorical(val_labels, 10)


# get new train data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28,28),
    batch_size=128,
    class_mode=None,
    shuffle=False
)

val_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(28,28),
    batch_size=128,
    class_mode=None,
    shuffle=False
)

x_train_new = []
x_val_new = []

for batch in train_generator:
    x_train_new.extend(batch)
x_train_new = np.array(x_train_new)

for batch in val_generator:
    x_val_new.extend(batch)
x_val_new = np.array(x_val_new)


x_train_concat = np.concatenate((x_train, x_train_new), axis=0)
x_val_concat = np.concatenate((x_val, x_val_new), axis=0)

# reshape data
x_train_concat = x_train_concat.reshape(x_train_concat.shape[0], 28, 28, 1)
x_val_concat = x_val_concat.reshape(x_val_concat.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# One-hot Encoding label
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# new label y
y_train_concat = np.concatenate((y_train, y_train_new), axis=0)
y_val_concat = np.concatenate((y_val, y_val_new), axis=0)


#defiantion model
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
#
print(model.summary())
#
# #Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, decay=0.0001),
              metrics=['accuracy'])

# Training Model
H = model.fit(x_train_concat, y_train_concat, validation_data=(x_val_concat, y_val_concat),
              batch_size=64, epochs=20, verbose=1)
#
# # Visualize loss, accuracy of training set and validation set
# fig = plt.figure()
# numOfEpoch = 10
# plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
# plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
# plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
# plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
# plt.title('Accuracy and Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss/Accuracy')
# plt.legend()
# plt.show()

model.save('MNIST_recognization.keras')

# if __name__=='__main__':
#     model = keras.models.load_model('MNIST_recognization.keras')
#
#     # Evaluate model on the test data
#     # score = model.evaluate(x_test, y_test, verbose=0)
#
#     # Predict image
#     # plt.imshow(x_test[0].reshape(28,28), cmap='gray')
#     # y_predict = model.predict(x_test[10].reshape(1,28,28,1))
#     # print(f'Predict value: {np.argmax(y_predict)}. Label: {y_test[10]}')
#
#     image_data = cv2.imread('Datasets/digit_8.jpg')
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
#     image_data = cv2.resize(image_data, (28,28))
#     image_data = image_data.reshape((28,28,1))
#
#     y_predict = model.predict(image_data.reshape(1,28,28,1))
#     print(f'Predict value: {y_predict}. Label: 8')

