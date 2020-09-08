#%% TBD (search is failing): Hyperparameter tuning for Multiclass classification
#reference https://github.com/keras-team/keras-tuner
# Restart the Spyder
import pandas as pd
import numpy as np
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc; gc.enable()

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

os.chdir("D:\\trainings\\tensorflow")

exec(open(os.path.abspath('tf_CommonUtils.py')).read())
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# Read data from https://www.kaggle.com/c/titanic/data
data = pd.read_csv("./data/kaggle_titanic_train_EncodedScaled.csv")
data.info()

LABEL = "Survived"
batch_size = 8

#Get list of independent features
ar_independent_features = np.setdiff1d(data.columns, LABEL)

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15)
training_set.shape
training_set.head(2)

len_fea = len(ar_independent_features)

# Build the model for tunning
def build_model_for_tunning(hp):
    model = tf.keras.models.Sequential() # same as tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=len_fea, max_value=10 * len_fea, step=len_fea), input_shape=(len_fea,), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=len_fea, max_value=10 * len_fea, step=len_fea), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    
    #Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    # def build_model_for_tunning

tuner = RandomSearch(
    build_model_for_tunning,
    objective='accuracy',
    max_trials=10,
    executions_per_trial=1, # to get results faster, set 1 (ideal 3) (single round of training for each model configuration).
    directory='./model/kaggle_titanic/',
    project_name='kaggle_titanic_tunning')

#print a summary of the search space:
tuner.search_space_summary()

# Train it
tuner.search(training_set[ar_independent_features].values, training_set[LABEL].values, epochs=10, batch_size=batch_size, use_multiprocessing= False)

#summary of the results
tuner.results_summary()

#best model
model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
model.evaluate(test_set[ar_independent_features].values, test_set[LABEL].values, verbose = 0)
# loss value & metrics values: [0.45, 0.79]

#Making Predictions
predictions = model.predict(x=test_set[ar_independent_features].values)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

#Statistics are also available as follows
print("Overall Accuracy is ", round(accuracy_score(test_set[LABEL].values, predictions_number), 2),", Kappa is ", round(abs(cohen_kappa_score(test_set[LABEL].values, predictions_number)), 2))
#Overall Accuracy is  0.76 , Kappa is  0.51

del(data, training_set, test_set, predictions, predictions_number); gc.collect()
