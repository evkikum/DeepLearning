#%% Transfer learning: Basic model
#Theory on ppt
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub # pretrained model components

from skimage.io import imread
from skimage.transform import resize

import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Get the link for module in steps: https://tfhub.dev/ -> module
classifier_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2'

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

im_wh = 224; channel = 3

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url,  input_shape=(im_wh,im_wh,channel)) # output_shape=[1001],
])

#Prepare test data
ar_image_files = np.array(['./data/images/balloon.jpg','./data/images/elephant.jpg','./data/images/forest.jpg'])
# image reading one by one

# image reading one by one
img = imread(ar_image_files[2]) # Change index for and see
img = resize(img, (im_wh, im_wh)) # tf.image.resize(img, (im_wh, im_wh))

plt.imshow(img)
plt.show()

#Making Predictions on images
predictions = classifier.predict(img[np.newaxis, ...])
predictions.shape

predictions_number = np.argmax(predictions[0], axis=-1)
predictions_number

plt.imshow(img)
plt.title("Prediction: " + imagenet_labels[predictions_number].title())
plt.show()

#Challenges: Make sure your data set is similar as source data used by model if not same then see next section
#CW: Play with few other type of images

#%% Transfer learning: Customise the model to recognize the classes in our dataset
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub # pretrained model components

from skimage.io import imread
from skimage.transform import resize

import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Few constants
folder_name = './data/mnist_image_data_generator'
folder_name_train = os.path.join(folder_name, 'train')
folder_name_eval = os.path.join(folder_name, 'eval')

im_wh = 224; channel = 3
batch_size = 32
epochs = 1

#These generators were created in previous section
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_data_gen = image_generator.flow_from_directory(folder_name_train, target_size=(im_wh, im_wh))
val_data_gen  = image_generator.flow_from_directory(folder_name_eval, target_size=(im_wh, im_wh))

#Get one batch just for learning else will take good amount of time in prediction
train_data, train_labels = next(train_data_gen)
val_data, val_labels = next(val_data_gen)

train_data.shape, train_labels.shape
val_data.shape, val_labels.shape

#See original data
plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(train_labels[i]))
    # end of 'for'
    
#get from TF2 compatible https://tfhub.dev/s?q=tf2
feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'
                        
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(im_wh, im_wh, channel))

#Freeze the variables in the feature extractor layer before compile and fit (train) the model
feature_extractor_layer.trainable = False

model = tf.keras.models.Sequential()
model.add(feature_extractor_layer)
model.add(tf.keras.layers.Dense(train_data_gen.num_classes, activation='softmax'))

model.summary() # See Trainable params and Non-trainable params

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data_gen, epochs=epochs, steps_per_epoch=20 * batch_size) # may take 15 min

ev = model.evaluate_generator(val_data_gen) #Will take 5 min
ev # [0.3, 0.93]

predictions = model.predict(train_data)

# Now see first one just to see the format
predictions[0]

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

#Confusion matrix is the right way to see the accuracy. We have learnt earlier
# and hence let us see only images to save time
    
plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    title = 'y ' if np.argmax(train_labels[i]) == predictions_number[i] else 'n '
    plt.xlabel(title + str(predictions_number[i]))
    # end of 'for'

#CW: The above prediction is not that great. How to improve

#%% Transfer learning: Use inbuilt (not from TF hub) model to recognize the classes in our dataset
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub # pretrained model components

from skimage.io import imread
from skimage.transform import resize

import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Few constants
folder_name = './data/mnist_image_data_generator'
folder_name_train = os.path.join(folder_name, 'train')
folder_name_eval = os.path.join(folder_name, 'eval')

im_wh = 160; channel = 3
batch_size = 32
epochs = 1

#These generators were created in previous section
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_data_gen = image_generator.flow_from_directory(folder_name_train, target_size=(im_wh, im_wh))
val_data_gen  = image_generator.flow_from_directory(folder_name_eval, target_size=(im_wh, im_wh))

#Get one batch just for learning else will take good amount of time in prediction
train_data, train_labels = next(train_data_gen)
val_data, val_labels = next(val_data_gen)

train_data.shape, train_labels.shape
val_data.shape, val_labels.shape

#See original data
plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(train_labels[i]))
    # end of 'for'

# https://www.tensorflow.org/api_docs/python/tf/keras/applications
# Create the base model from the pre-trained model MobileNet V2
model_pretrained = tf.keras.applications.MobileNetV2(input_shape=(im_wh, im_wh, channel), include_top=False, weights='imagenet')
model_pretrained.trainable = False

model_pretrained.summary() # Obsreve how big model it is

model = tf.keras.models.Sequential()
model.add(model_pretrained)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # See Trainable params and Non-trainable params

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data_gen, epochs=epochs, steps_per_epoch=20 * batch_size) # may take 15 min

ev = model.evaluate_generator(val_data_gen) #Will take 5 min
ev # [0.3, 0.93]

predictions = model.predict(train_data)

# Now see first one just to see the format
predictions[0]

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

#Confusion matrix is the right way to see the accuracy. We have learnt earlier
# and hence let us see only images to save time

plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    title = 'y ' if np.argmax(train_labels[i]) == predictions_number[i] else 'n '
    plt.xlabel(title + str(predictions_number[i]))
    # end of 'for'

#%% Transfer learning: Customise inbuilt (not from TF hub) model to recognize the classes in our dataset
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub # pretrained model components

from skimage.io import imread
from skimage.transform import resize

import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Few constants
folder_name = './data/mnist_image_data_generator'
folder_name_train = os.path.join(folder_name, 'train')
folder_name_eval = os.path.join(folder_name, 'eval')

im_wh = 160; channel = 3
batch_size = 32
epochs = 1

#These generators were created in previous section
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_data_gen = image_generator.flow_from_directory(folder_name_train, target_size=(im_wh, im_wh))
val_data_gen  = image_generator.flow_from_directory(folder_name_eval, target_size=(im_wh, im_wh))

#Get one batch just for learning else will take good amount of time in prediction
train_data, train_labels = next(train_data_gen)
val_data, val_labels = next(val_data_gen)

train_data.shape, train_labels.shape
val_data.shape, val_labels.shape

#See original data
plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(train_labels[i]))
    # end of 'for'

# https://www.tensorflow.org/api_docs/python/tf/keras/applications
# Create the base model from the pre-trained model MobileNet V2
model_pretrained = tf.keras.applications.MobileNetV2(input_shape=(im_wh, im_wh, channel), include_top=False, weights='imagenet')

#Make true to all
model_pretrained.trainable = True

len(model_pretrained.layers)

# Freeze all the layers except last 5
for layer in model_pretrained.layers[:-5]:
    layer.trainable =  False
  
model_pretrained.summary() # Obsreve how big model it is

model = tf.keras.models.Sequential()
model.add(model_pretrained)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() # See Trainable params and Non-trainable params

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_data_gen, epochs=epochs, steps_per_epoch=20 * batch_size) # may take 100 min

ev = model.evaluate_generator(val_data_gen) #Will take 5 min
ev # [0.3, 0.93]

predictions = model.predict(train_data)

# Now see first one just to see the format
predictions[0]

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

#Confusion matrix is the right way to see the accuracy. We have learnt earlier
# and hence let us see only images to save time

plt.figure(figsize=(10,10))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    title = 'y ' if np.argmax(train_labels[i]) == predictions_number[i] else 'n '
    plt.xlabel(title + str(predictions_number[i]))
    # end of 'for'

#Not that good, may require more training
    
del(seed, folder_name, folder_name_train, folder_name_eval, im_wh, channel, batch_size, epochs, image_generator, train_data_gen, val_data_gen, train_data, train_labels, val_data, val_labels, model_pretrained, model, predictions, predictions_number)
