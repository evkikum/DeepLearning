#%% Building a Convolutional Neural Network for Image (MNIST) classification
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

#List of availble TF dataset https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/datasets?hl=en
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

#Reshape data to add channel as per need of CNN - (count of images,image_height, image_width, color_channels)
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
model.fit(train_data, train_labels, epochs=5, workers=4, use_multiprocessing=True)
#batch_size=100

ev = model.evaluate(eval_data, eval_labels, verbose = 0)
ev #[0.024, 0.993]

#Making Predictions
predictions = model.predict(eval_data)

# Now see first one just to see the format
predictions[0]

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(eval_labels, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# 5: Overall Accuracy is  0.99 , Kappa is  0.99

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(train_data, train_labels, eval_data, eval_labels, im_wh, model, ev, predictions, predictions_number, confusion_matrix, df, cms)
#%% CW: do the above with data tf.keras.datasets.fashion_mnist

#%% CapsuleNets (TBD: full code to be done. Incomplete now)
# How to install
# 1. git clone https://github.com/XifengGuo/CapsNet-Keras.git capsnet-keras
# 2. move 'capsnet-keras' folder to D:\ProgramFiles\Anaconda3\Lib\site-packages
# 3. rename folder to 'capsnet_keras'

#from capsnet_keras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

#%% Building a CNN with ImageDataGenerator
#Theory on ppt 
import numpy as np
import tensorflow as tf
import os
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

folder_name = './data/mnist_image_data_generator'

# Load training and eval data
(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

save_in_image_data_generator_format(folder_name, train_data, train_labels, eval_data, eval_labels)
del(train_data, train_labels, eval_data, eval_labels)

#Above is ONE time effort. Now let use for actual classifications purpose

#Few constants
folder_name_train = os.path.join(folder_name, 'train')
folder_name_eval = os.path.join(folder_name, 'eval')

batch_size = 32
epochs = 5
im_wh = 28

#Now assume that you are starting from here and no knowledge of data in folders

#It can do a lot of things - read images, scale ...
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # Generator for our training data
#validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # Generator for our validation data

#Now, load images from the disk, scale, and resizes
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=folder_name_train, color_mode="grayscale", shuffle=True, target_size=(im_wh, im_wh), class_mode='sparse')
val_data_gen  = train_image_generator.flow_from_directory(batch_size=batch_size, directory=folder_name_eval, color_mode="grayscale", target_size=(im_wh, im_wh), class_mode='sparse')
#Visualize: The next function returns a batch from the dataset in form of (x_train, y_train)

train_data, train_labels = next(train_data_gen)

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

del(train_data, train_labels)

#Create model similar to above
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(im_wh, im_wh, 1))) #, padding="same"
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # Note shape of images is going down. Note: it'll not go down when padding="same"

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data_gen,
    steps_per_epoch= 20 * batch_size, #if know total count then use tot_count/batch_size
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=20 * batch_size # ,workers=4, use_multiprocessing=True 
    )
# Oserve, it has loaded 20 * batch_size in one step

ev = model.evaluate_generator(val_data_gen)
ev # [0.03, 0.99]

#Making Predictions
predictions = model.predict_generator(val_data_gen)

# Now see first one just to see the format
predictions[0]

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Just precaution
predictions_number = predictions_number.astype(int)

#Let us play with various features
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=45, horizontal_flip=True, zoom_range=0.5)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=folder_name_train, color_mode="grayscale", shuffle=True, target_size=(im_wh, im_wh), class_mode='sparse')

train_data, train_labels = next(train_data_gen)

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

#Clean
del(train_data, train_labels, folder_name, folder_name_train, folder_name_eval, batch_size, epochs, im_wh, train_image_generator, train_data_gen, val_data_gen, model, ev, predictions, predictions_number)
