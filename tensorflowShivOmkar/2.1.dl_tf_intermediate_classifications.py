#%% Binary classification On Kaggle data using Tensorflow Multi level
# Restart the Spyder
import pandas as pd
import numpy as np
import tensorflow as tf
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
data = pd.read_csv("./data/kaggle_titanic_train.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe())
#print(data.describe(include = [np.number])) # for number only

#Drop few columns, may not be use ful for current analysis
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data.info() # Now see if any missing values in any columns

#Note: how to impute missing value is purview of ML. Here, do simple thing so that we focus on DL
age_avg = data['Age'].mean()
data['Age'].fillna(value = age_avg, inplace=True)

#Now, drop rows if any missing
data.dropna(inplace=True)

data.info() # Now see if any missing values in any columns

# identifications of features and response. Detail'll be explained in a few minutes
NUM_FEATURES = ['Pclass','SibSp','Parch','Fare']
bucketized_FEATURES = 'Age'
categorical_FEATURES = 'Sex'
embedding_FEATURES = 'Embarked'
crossed_FEATURES = 'Embarked' # With Age

FEATURES = np.append(np.append(np.append(np.append(NUM_FEATURES, bucketized_FEATURES), categorical_FEATURES), embedding_FEATURES), crossed_FEATURES)
FEATURES = np.unique(FEATURES)

LABEL = "Survived"
batch_size = 8

#Do the data type conversion for category
data[[categorical_FEATURES,embedding_FEATURES]] = data[[categorical_FEATURES,embedding_FEATURES]].apply(lambda x: x.astype('category'))
data.dtypes

#One hot encode
data = Encoding(data, LABEL, scale_and_center = True, fileTrain = "./data/kaggle_titanic_train_EncodedScaled.csv")
data.head(2)

#Get list of independent features
ar_independent_features = np.setdiff1d(data.columns, LABEL)

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15)
del(data)
training_set.shape
training_set.head(2)

len_fea = len(ar_independent_features)

# Build the model
model = tf.keras.models.Sequential() # same as tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2*len_fea, input_shape=(len_fea,), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(len_fea, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.summary()

#Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train it
model.fit(training_set[ar_independent_features].values, training_set[LABEL].values, epochs=100, batch_size=batch_size) # 6 min

#Save and retrieve
model.save('./model/model_tf_kaggle_titanic_binary_classsification.h5')
#model = tf.keras.models.load_model('./model/model_tf_kaggle_titanic_binary_classsification.h5')

# Evaluate on test data
model.evaluate(test_set[ar_independent_features].values, test_set[LABEL].values, verbose = 0)
# loss value & metrics values: [0.45, 0.79]

#Making Predictions
predictions = model.predict(x=test_set[ar_independent_features].values)

# Extracting max probability
predictions_number = np.array([])
for row_num in range(predictions.shape[0]): # row_num = 0
    predictions_number = np.append(predictions_number, np.argmax(predictions[row_num]))

#Few statistics
confusion_matrix(test_set[LABEL].values, predictions_number)
classification_report(test_set[LABEL].values, predictions_number)

#Statistics are also available as follows
print("Overall Accuracy is ", round(accuracy_score(test_set[LABEL].values, predictions_number), 2),", Kappa is ", round(abs(cohen_kappa_score(test_set[LABEL].values, predictions_number)), 2))
#Overall Accuracy is  0.81 , Kappa is  0.56

del(training_set, test_set, predictions_number); gc.collect()

#%% Binary classification: Explore few more ways to better classification
# Restart the Spyder
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import gc; gc.enable()

tf.keras.backend.clear_session()  # For easy reset of notebook state

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('precision', 2)

os.chdir("D:\\trainings\\tensorflow")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

# Read data
data = pd.read_csv("./data/kaggle_titanic_train.csv")
data.shape
data.dtypes
data.head(2)
data.info()
print(data.describe())
#print(data.describe(include = [np.number])) # for number only

#Drop few columns, may not be use ful for current analysis
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data.info() # Now see if any missing values in any columns

#Note: how to impute missing value is purview of ML. Here, do simple thing so that we focus on DL
age_avg = data['Age'].mean()
data['Age'].fillna(value = age_avg, inplace=True)

#Now, drop rows if any missing
data.dropna(inplace=True)

data.info() # Now see if any missing values in any columns

# identifications of features and response. Detail'll be explained in a few minutes
NUM_FEATURES = ['Pclass','SibSp','Parch','Fare']
bucketized_FEATURES = 'Age'
categorical_FEATURES = 'Sex'
embedding_FEATURES = 'Embarked'
crossed_FEATURES = 'Embarked' # With Age

FEATURES = np.append(np.append(np.append(np.append(NUM_FEATURES, bucketized_FEATURES), categorical_FEATURES), embedding_FEATURES), crossed_FEATURES)
FEATURES = np.unique(FEATURES)

LABEL = "Survived"
batch_size = 8

#Do the data type conversion for category
data[[categorical_FEATURES,embedding_FEATURES]] = data[[categorical_FEATURES,embedding_FEATURES]].apply(lambda x: x.astype('category'))

#Segragate 85% and 15%
training_set ,test_set = train_test_split(data,test_size=0.15, random_state = seed, stratify = data[LABEL])

#Building the input_fn: regressor accepts Tensors and custom function to convert pandas
#Dataframe and return feature column and label values as Tensors:
def input_fn(features, labels = None, custom_batch_size = batch_size, caller_source = 'train'):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    
    if caller_source != 'test':
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    if caller_source == 'train': 
        dataset = dataset.shuffle(len(features)) #if ".repeat()" is added here then add "epochs steps_per_epoch" in fit
        
    dataset = dataset.batch(custom_batch_size)

    return dataset

#train in iterable dataset
ds_train = input_fn(training_set[FEATURES], training_set[LABEL],custom_batch_size = batch_size)

#Create feature columns
feature_cols = []

# numeric cols
for num_col in NUM_FEATURES:
  feature_cols.append(tf.feature_column.numeric_column(num_col, dtype=tf.float32))

#bucketized cols: If don't want to feed a number directly odel, but instead split its value into
#different categories based on numerical ranges. 
#Buckets include the left boundary, and exclude the right boundary. 
bucketized_col = tf.feature_column.numeric_column(bucketized_FEATURES, dtype=tf.float32)
age_buckets = tf.feature_column.bucketized_column(bucketized_col, boundaries=[30, 40, 50, 60])
feature_cols.append(age_buckets)

# indicator cols
cat_vocab = tf.feature_column.categorical_column_with_vocabulary_list(categorical_FEATURES, pd.unique(data[categorical_FEATURES].values))
cat_one_hot = tf.feature_column.indicator_column(cat_vocab)
feature_cols.append(cat_one_hot)

# Just to see - one hot encoding
first_batch = next(iter(ds_train))[0]
feature_layer = tf.keras.layers.DenseFeatures(cat_one_hot)
print(feature_layer(first_batch).numpy())
  
#Embedding cols: When there are large values per category then use an embedding column to 
#overcome this limitation. Instead of representing the data as a one-hot vector of many 
#dimensions, an embedding column represents that data as a lower-dimensional, dense vector in 
#which each cell can contain any number, not just 0 or 1. The size of the embedding (8, in the 
#example below) is a parameter that must be tuned.
embedding_col = tf.feature_column.embedding_column(cat_vocab, dimension=8) # 8 Need to be tuned
feature_cols.append(embedding_col)

# Just to see - one hot encoding
first_batch = next(iter(ds_train))[0]
feature_layer = tf.keras.layers.DenseFeatures(embedding_col)
print(feature_layer(first_batch).numpy())

#CW: Read 'Hashed feature columns' and practice above

## crossed cols TBD: Not working
#cat_vocab_crosssed = tf.feature_column.categorical_column_with_vocabulary_list(crossed_FEATURES, pd.unique(data[crossed_FEATURES].values))
#crossed_feature = tf.feature_column.crossed_column([age_buckets, cat_vocab_crosssed], 
#                                                   hash_bucket_size=10) # Max size of all combination
#crossed_feature = tf.feature_column.indicator_column(crossed_feature)
#feature_cols.append(crossed_feature)

# Model in this way
feature_cols = tf.keras.layers.DenseFeatures(feature_cols)

# Build the model
model = tf.keras.models.Sequential() # same as tf.keras.Sequential()
model.add(feature_cols)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

#Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

model.summary()

model.fit(ds_train, epochs = 100) # epochs steps_per_epoch=100,
#Because of embedding is shows 95 (755/8)

# Evaluate on test data
test_ds = input_fn(test_set[FEATURES], test_set[LABEL],custom_batch_size = batch_size, caller_source = 'eval')
model.evaluate(test_ds)
# loss value & metrics values: [0.13, 0.95]

#Making Predictions
test_ds = input_fn(test_set[FEATURES], test_set[LABEL],custom_batch_size = batch_size, caller_source = 'test')
predictions = model.predict(test_ds, verbose=1)
predictions = np.ravel(predictions)

#Explain why cutoff is required and why not 0.5 always works
cutoff = data[data[LABEL] == 1].shape[0]/data.shape[0]
predictions_number = tf.where(predictions >= (1-cutoff), 1, 0)

#Few statistics
confusion_matrix(test_set[LABEL].values, predictions_number)
classification_report(test_set[LABEL].values, predictions_number)

#Statistics are also available as follows
print("Overall Accuracy is ", round(accuracy_score(test_set[LABEL].values, predictions_number), 2),", Kappa is ", round(abs(cohen_kappa_score(test_set[LABEL].values, predictions_number)), 2))
#Overall Accuracy is  0.81 , Kappa is  0.58
# Compare with 'without embedding' done above. even though number of records come down, not much difference in accuracy

del(data, training_set, test_set, predictions_number, age_avg, NUM_FEATURES, bucketized_FEATURES, categorical_FEATURES, embedding_FEATURES, crossed_FEATURES, FEATURES, LABEL, batch_size, ds_train, feature_cols, bucketized_col, age_buckets, cat_vocab, cat_one_hot, first_batch, feature_layer, embedding_col, model, test_ds, cutoff); gc.collect()

## Which one to use?
## Say - you have 10 million records although you can process 1 million in memory