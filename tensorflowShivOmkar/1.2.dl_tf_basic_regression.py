#%% Building Regression with Premade Estimators
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.random.set_seed(seed)

#read data. Details is at https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
data = pd.read_csv("./data/boston.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe()) # .unstack()
#print(data.describe(include = [np.number])) # for number only

# identifications of features and response
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"
batch_size = 8

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
    
    if caller_source == 'train': #test
        dataset = dataset.shuffle(len(features)).repeat()
        
    dataset = dataset.batch(custom_batch_size)

    return dataset

#Feature column describs how the model should use raw input data from the features dictionary.
#All features in data set contain continuous values, hence create their FeatureColumn

#Defining FeatureColumns and Creating the Regressor
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Clean model folder from previous models as new model will overide or append depending upon settings
model_dir="./log/boston_model/"; # remove_nonempty_folder(model_dir); os.makedirs(model_dir)

# instantiate a DNNRegressor for the neural network regression model
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[20, 20], model_dir=model_dir)

#Training the Regressor 
regressor.train(input_fn=lambda: input_fn(training_set[FEATURES], training_set[LABEL],custom_batch_size = batch_size), steps=5000)

#Evaluating the Model
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set[FEATURES], test_set[LABEL],custom_batch_size = batch_size, caller_source = 'eval'))

#Retrieve the loss from the ev results and print it to output:
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

#Making Predictions
y = regressor.predict(input_fn=lambda: input_fn(test_set[FEATURES], None, custom_batch_size = batch_size, caller_source = 'test'))

# .predict() returns an iterator; convert to a list and print predictions
predictions = list(pred_tensor["predictions"][0] for pred_tensor in itertools.islice(y, test_set.shape[0]))
print ("Predictions: {}".format(str(predictions)))

#RMSE
rmse = np.sqrt(mean_squared_error(test_set[LABEL], predictions))
print(rmse) # batch 8: 9.75

# To view tensorboard, go to anaconda prompt -> respective model_dir
# tensorboard --logdir=.

# Cleaning
del(training_set, test_set, predictions, feature_cols, ev, loss_score, rmse, model_dir, batch_size); gc.collect()

#%% Regression by using tf.keras model layers
# Restart the Spyder
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")
#exec(open(os.path.abspath('tf_CommonUtils.py')).read())
# identifications of features and response
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# load dataset
data = pd.read_csv("./data/boston.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe()) # .unstack()
#print(data.describe(include = [np.number])) # for number only

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15)
del(data)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(len(FEATURES), input_shape=(len(FEATURES),), activation=tf.nn.relu)) # , kernel_initializer = tf.random_normal_initializer
model.add(tf.keras.layers.Dense(1)) # , kernel_initializer = tf.random_normal_initializer

#Compile
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

model.summary()

# Train it
model.fit(training_set[FEATURES].values, training_set[LABEL].values, epochs=100,batch_size=8, shuffle=True)

# Evaluate on test data
model.evaluate(test_set[FEATURES].values, test_set[LABEL].values) # [81, 81]

#Making Predictions
predictions = model.predict(x=test_set[FEATURES].values, verbose=1)
predictions = np.ravel(predictions)

#RMSE
rmse = np.sqrt(mean_squared_error(test_set[LABEL].values, predictions))
print("RMSE: ", rmse) # 7.59
