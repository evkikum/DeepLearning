#%% First standard without any optimization Technique
import numpy as np
import tensorflow as tf
import os
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)


#List of avalble TF dataset https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/datasets?hl=en
#Details is at http://yann.lecun.com/exdb/mnist/
#https://github.com/petar/GoMNIST/tree/master/data

# Load training and eval data
(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()
train_data.shape # (60000, 28, 28)
eval_data.shape # (10000, 28, 28)

train_labels
eval_labels

#constants
im_wh = train_data.shape[1] # Assuming that width and height are same else make by transformation

#View the first 10 images and the class name
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    x_image = np.reshape(train_data[i], [im_wh, im_wh])
    plt.imshow(x_image, cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
    # end of 'for'

#Reshape data to add channel as per need of CNN - (image_height, image_width, color_channels)
train_data = train_data.reshape((train_data.shape[0], im_wh, im_wh, 1))
eval_data = eval_data.reshape((eval_data.shape[0], im_wh, im_wh, 1))

#Scale these values to a range of 0 to 1
train_data = train_data / 255.0
eval_data = eval_data / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1))) #, padding="same"
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to keep time short, set epochs = 1
model.fit(train_data, train_labels, epochs=1, workers=4, use_multiprocessing=True)

#Making Predictions
predictions = model.predict(eval_data)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# none: Overall Accuracy is  0.99 , Kappa is  0.99


#%% Weight regularization: L2 (weight decay) -> cost added is proportional to the square of the value of the weights coefficients
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1), kernel_regularizer= tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer= tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to keep time short, set epochs = 1
model.fit(train_data, train_labels, epochs=1, workers=4, use_multiprocessing=True)

#Making Predictions
predictions = model.predict(eval_data)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# L2: Overall Accuracy is  0.98 , Kappa is  0.98
# none: Overall Accuracy is  0.99 , Kappa is  0.99

#%% Weight regularization: L1 -> cost added is proportional to the absolute value of the weights coefficients
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1), kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to keep time short, set epochs = 1
model.fit(train_data, train_labels, epochs=1, workers=4, use_multiprocessing=True)

#Making Predictions
predictions = model.predict(eval_data)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# L1: Overall Accuracy is  0.97 , Kappa is  0.96
# L2: Overall Accuracy is  0.98 , Kappa is  0.98
# none: Overall Accuracy is  0.99 , Kappa is  0.99

#%% Dropout
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25)) # https://stats.stackexchange.com/questions/147850/are-pooling-layers-added-before-or-after-dropout-layers
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to keep time short, set epochs = 1
model.fit(train_data, train_labels, epochs=1, workers=4, use_multiprocessing=True)

#Making Predictions
predictions = model.predict(eval_data)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# Dropout: Overall Accuracy is  0.99 , Kappa is  0.99
# L1: Overall Accuracy is  0.97 , Kappa is  0.96
# L2: Overall Accuracy is  0.98 , Kappa is  0.98
# none: Overall Accuracy is  0.99 , Kappa is  0.99

#%% Batch Normalisation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(im_wh, im_wh, 1))) # , activation='relu'
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU()) # It is advisable to add activation after BN
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU()) # It is advisable to add activation after BN
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU()) # It is advisable to add activation after BN
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# to keep time short, set epochs = 1
model.fit(train_data, train_labels, epochs=1, workers=4, use_multiprocessing=True)

#Making Predictions
predictions = model.predict(eval_data)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# Batch Normalisation: Overall Accuracy is  0.99 , Kappa is  0.99
# Dropout: Overall Accuracy is  0.99 , Kappa is  0.99
# L1: Overall Accuracy is  0.97 , Kappa is  0.96
# L2: Overall Accuracy is  0.98 , Kappa is  0.98
# none: Overall Accuracy is  0.99 , Kappa is  0.99

#%% EarlyStopping
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1))) #, padding="same"
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Condition, when to early stop
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)

# to keep time short, set epochs = 1 & [:5000]
model.fit(train_data[:5000], train_labels[:5000], epochs=100, workers=4, use_multiprocessing=True, callbacks=[early_stop])

