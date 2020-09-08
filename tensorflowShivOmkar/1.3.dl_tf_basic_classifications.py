#%% Classifications using Premade Estimators
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
import itertools
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# identifications of features and response
FEATURES = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
LABEL = "Species"
batch_size = 8

data = pd.read_csv("./data/iris.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe()) # .unstack()
#print(data.describe(include = [np.number])) # for number only

# Labels dtype should be integer
data[LABEL].unique() #.astype(int)
num_mapping = {"setosa":0 ,"versicolor" :1,"virginica":2}
data[LABEL] = data[LABEL].replace(num_mapping)

# now convert the types
data[LABEL] = pd.to_numeric(data[LABEL], errors='coerce')
data.dtypes

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15)
del(data)

#An input function returns a tf.data.Dataset object which contains features (dictionary -
#with key (feature name) and value (feature's values)
#label - An array containing the values of the label for every row.

#Building the input_fn: regressor accepts Tensors and custom function to convert pandas
#Dataframe and return feature column and label values as Tensors:
def input_fn(features, labels = None, custom_batch_size = batch_size, caller_source = 'train'):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    
    if caller_source != 'test':
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    if caller_source == 'train': 
        dataset = dataset.shuffle(len(features)).repeat()
        
    dataset = dataset.batch(custom_batch_size)

    return dataset

#Defining FeatureColumns and Creating the classifier
#All features in data set contain continuous values, hence create their FeatureColumns
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols, hidden_units=[30,10], n_classes=3)

# Fit model. Note: If error comes then clean folder 'model_dir' and restart Spyder
classifier.train(input_fn=lambda: input_fn(training_set[FEATURES], training_set[LABEL],custom_batch_size = batch_size), steps=5000)

#Evaluating the Model. Note: If error comes then clean folder 'model_dir' and restart Spyder
ev = classifier.evaluate(input_fn=lambda: input_fn(test_set[FEATURES], test_set[LABEL],custom_batch_size = batch_size, caller_source = 'eval'))
print("\nTest Accuracy: {0:f}\n".format(ev["accuracy"])) # 78%

#Making Predictions
predictions = classifier.predict(input_fn=lambda: input_fn(test_set[FEATURES], None, custom_batch_size = batch_size, caller_source = 'test'))

# .predict() returns an iterator; convert to a list and print predictions
predictions = list(pred_tensor["class_ids"][0] for pred_tensor in itertools.islice(predictions, test_set.shape[0]))
print ("Predictions: {}".format(str(predictions)))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(test_set[LABEL].values, predictions)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#Overall Accuracy is  0.78 , Kappa is  0.66

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

# Cleaning
del(training_set, test_set, predictions, feature_cols, df, cms, ev, num_mapping, batch_size); gc.collect()

#%% Multiclass classification using Tensorflow Multi level
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# identifications of features and response
FEATURES = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
LABEL = "Species"

# Read data
data = pd.read_csv("./data/iris.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe()) # .unstack()
#print(data.describe(include = [np.number])) # for number only

# Labels dtype should be integer
data[LABEL].unique() #.astype(int)
num_mapping = {"setosa":0 ,"versicolor" :1,"virginica":2}
data[LABEL] = data[LABEL].replace(num_mapping)

# now convert the types
data[LABEL] = pd.to_numeric(data[LABEL], errors='coerce')
data.dtypes

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15)
del(data)

# Build the model
model = tf.keras.models.Sequential() # same as tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(4,), activation=tf.nn.relu)) # , kernel_initializer = tf.random_normal_initializer
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax)) # , kernel_initializer = tf.random_normal_initializer

model.summary()

#Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train it
model.fit(training_set[FEATURES].values, training_set[LABEL].values, epochs=1000,batch_size=8) # 6 min

# Evaluate on test data
model.evaluate(test_set[FEATURES].values, test_set[LABEL].values)
# loss value & metrics values: [0.13, 0.95]

#Making Predictions
predictions = model.predict(x=test_set[FEATURES].values, verbose=1)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(test_set[LABEL].values, predictions_number)
confusion_matrix

# normalized confusion matrix
confusion_matrix.plot(normalized=True)
plt.show()

#Statistics are also available as follows
confusion_matrix.print_stats()
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# 1000: Overall Accuracy is  0.96 , Kappa is  0.93

df = cms['class'].reset_index()
df[df['index'].str.contains('Precision')]
df[df['index'].str.contains('Sensitivity')]
df[df['index'].str.contains('Specificity')]

del(training_set, test_set, predictions, df, cms, num_mapping, predictions_number); gc.collect()
