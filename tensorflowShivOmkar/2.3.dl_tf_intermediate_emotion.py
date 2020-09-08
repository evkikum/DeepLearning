#%% Emotion recognition with CNNs
# download data from https://inclass.kaggle.com/c/facial-keypoints-detector/data
#input: 48x48 pixel gray values (between 0 and 255)
#target: emotion category (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)

import tensorflow as tf

import numpy as np
import os
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# few constants, Image related
IMAGE_SIZE = 48; NUM_LABELS = 7; VALIDATION_PERCENT = 0.1  # use 10 percent of training images for validation
IMAGE_LOCATION_NORM = IMAGE_SIZE / 2
dict_labels = {0:'anger', 1:'disgust', 2: 'fear', 3:'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# It reads emotion data
# data_dir = './data/'; train_filename = "emotion_train.csv"; test_filename = "emotion_test.csv"
def read_emotion_data(data_dir, train_filename, test_filename):
    train_filename = os.path.join(data_dir, train_filename)
    data_frame = pd.read_csv(train_filename)
    data_frame.head(2)

    # Extract image data
    data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
    data_frame = data_frame.dropna()

    train_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    train_images.shape

    # Extract image label data
    train_labels = tf.keras.utils.to_categorical(data_frame['Emotion'], NUM_LABELS)
    #train_labels = data_frame['Emotion'].apply(create_onehot_label)
    #train_labels = np.vstack(train_labels).reshape(-1, NUM_LABELS)
    train_labels.shape
    train_labels[0]

    # Shuffle for better training and validation samples
    permutations = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutations]
    train_labels = train_labels[permutations]

    # Extract train and validation samples
    validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
    validation_images = train_images[:validation_percent]
    validation_labels = train_labels[:validation_percent]
    train_images = train_images[validation_percent:]
    train_labels = train_labels[validation_percent:]

    # Reading test.csv
    test_filename = os.path.join(data_dir, test_filename)
    data_frame = pd.read_csv(test_filename)
    data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
    data_frame = data_frame.dropna()
    test_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return train_images, train_labels, validation_images, validation_labels, test_images
# end of read_emotion_data

#Load the data
train_data, train_labels, eval_data, eval_labels, test_images = read_emotion_data('./data/', "emotion_train.csv", "emotion_test.csv")
train_data.shape, train_labels.shape, eval_data.shape, eval_labels.shape, test_images.shape
# ((3761, 48, 48, 1), (3761, 7), (417, 48, 48, 1), (417, 7), (1312, 48, 48, 1))

#Display the first image of the training set and its correct label:
image_seq_temp = 5
image_0 = train_data[image_seq_temp]
print(train_labels[image_seq_temp]), print(np.argmax(train_labels[image_seq_temp]))
image_0.shape
image_0 = np.resize(image_0,(48,48))

# Plot
plt.imshow(image_0, cmap='Greys_r')
plt.show()

#View the first 10 images and the class name
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_0 = train_data[i]
    x_image = np.reshape(image_0, [48, 48])
    plt.imshow(x_image, cmap=plt.cm.binary)
    plt.xlabel(dict_labels[np.argmax(train_labels[i])])
    # end of 'for'

# Convert the data to numeric
train_data = train_data.astype('float32')
eval_data = eval_data.astype('float32')

print('Train samples: ', train_data.shape[0],', Train data shape: ', train_data.shape)
print('Test samples: ', eval_data.shape[0])

#Train a CNN model
model = tf.keras.models.Sequential()

#The input image (48 pixel) is processed in the first convolutional layer using
#5x5 convolutional kernels. This results in 32 features, one for each filter used.
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]))) # kernel: length of the convolution window.
model.add(tf.keras.layers.BatchNormalization())

#The images are also downsampled by a maxpooling operation, to decrease the
#images from 48x48 to 24x24 pixels.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#These 32 smaller images are then processed by a second convolutional layer;
#this results in 64 new features
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

#resulting images are downsampled again to 12x12 pixels, by a second pooling operation.
#The output of this second pooling layer is formed by 64 images of 12x12 pixels each.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

#These are then flattened to a single vector of length 12x12x64 = 9,126, which
#is used as the input to a fully-connected layer with 256 neurons.
model.add(tf.keras.layers.Flatten())
#model.add(Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256,use_bias=False)) #13
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu')) #14
model.add(tf.keras.layers.Dropout(0.25))

#This feeds into another fully-connected layer with 'NUM_LABELS'  neurons classes
model.add(tf.keras.layers.Dense(NUM_LABELS, activation='softmax'))

# Define loss function, accuracy metrices
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.001),  metrics=['accuracy'])

#Now, we can fit the model. This should take about 10-15 seconds per epoch on a commodity GPU, or about 2-3 minutes for 12 epochs.
batch_size = 32
epochs = 12 #12 will take 36 min

# Train the model
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(eval_data, eval_labels))

#Evaluate the model
score = model.evaluate(eval_data, eval_labels, verbose=1)
print('Eval loss:', score[0]) # .60
print('Eval accuracy:', score[1]) # 0.81

#Making Predictions
predictions = model.predict(x=eval_data)

# See the probability
predictions[0] # you can see np.argmax(predictions[0])
eval_labels[0]

# Extract (non hot code to simple code) the predictions
predictions_number = np.array([]); actual_number = np.array([])
for row_num in range(eval_data.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))
    actual_number = np.append(actual_number, np.argmax(eval_labels[row_num]))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(actual_number, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# Epoch 12: Overall Accuracy is  0.81 , Kappa is  0.77

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

#Testing the model on your own image
#The dataset we use is standardized. All faces are pointed exactly at the camera
#and the emotional expressions are exaggerated, and even comical in some situations.
#Now let's see what happens if we use a more natural image. First, we need to make
#sure that that there is no text overlaid on the face, the emotion is recognizable,
#and the face is pointed mostly at the camera.

from skimage.io import imread
# from skimage.color import rgb2gray

# Gray image reading. It normalise before returning
img = imread('./data/images/sample_48p.jpg', as_gray=True)
img.shape # H x W x C
img.size # multiplication of WxHxC
type(img)

plt.imshow(img)
plt.show()

#Making Predictions on live image
pred = model.predict(x=img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1))
dict_labels[np.argmax(pred)]

# Gray image (48x48) reading. It normalise before returning
img = imread('./data/images/shiv_48p.jpg', as_grey=True)
img.shape # H x W x C
img.size # multiplication of WxHxC
type(img)

# See it
plt.imshow(img)
plt.show()

#Making Predictions on live image
pred = model.predict(x=img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1))
dict_labels[np.argmax(pred)]
