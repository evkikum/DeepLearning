#%% LSTM using Keras -Univariate Time Series -> working with train-test mode
#The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained
#using Backpropagation Through Time and overcomes the vanishing gradient problem.
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc; gc.enable()
os.chdir("D:\\trainings\\tensorflow")

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Now, we can load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('./data/AirPassengers.csv')

#date time conversion
data['TravelDate'] =  pd.to_datetime(data['TravelDate'], format='%m/%d/%Y')
data['Passengers'] = data['Passengers'].astype(float)
data.dtypes # Notice the dtype=’datetime[ns]’

#Index: getting time to index
data.set_index('TravelDate', inplace=True)
data.head()

#Convert data into array that can be broken up into training "batches" that we will feed into our
#RNN model.
TS = data['Passengers'] # This format is useful in plotting graph
TS.head(2)
TS.plot()

dataset = np.array(TS)
dataset = dataset.astype('float32')

#LSTMs are sensitive to the scale of the input data, specifically when the sigmoid or
# tanh (default) activation functions are used. It can be a good practice to rescale
#the data to the range of 0 to 1,also called normalizing.

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = dataset.reshape(-1, 1) # Follwoing function need in this shape else deprecation warning
np.min(np.ravel(dataset)), np.max(np.ravel(dataset))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
type(train), type(test)

#Now we can define a function to create a new dataset, as described above.
#The function takes two arguments: the dataset, which is a NumPy array that we want to convert into a
# dataset, and the look_back, which is the number of previous time steps to use as input variables to
# predict the next time period — in this case defaulted to 1.

#This default will create a dataset where X is the number of passengers at a given time (t) and Y is
#the number of passengers at the next time (t + 1).
#It can be configured, and we will by constructing a differently shaped dataset in the next section.

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#Let’s take a look at the effect of this function on the first rows of the dataset (shown in the
#unnormalized form for clarity).

# reshape into X=t and Y=t+1 using above function
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX.shape,trainY.shape,testX.shape,testY.shape
trainX[0:look_back], trainY[0:look_back]

#The LSTM network expects the input data (X) to be provided with a specific array structure in the form
#of: [samples, time steps, features].
#Currently, our data is in the form: [samples, features] and we are framing the problem as one time
#step for each sample. We can transform the prepared train and test input data into the expected
#structure using numpy.reshape() as follows:

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
trainX.shape

#We are now ready to design and fit our LSTM network for this problem.
#The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks, and
# an output layer that makes a single value prediction. The default 'tanh'
#activation function is used for the LSTM blocks. The network is trained for 100
#epochs and a batch size of 1 is used.

# create and fit the LSTM network
model = tf.keras.models.Sequential()
layers = [1, 50, 100, 1]
model.add(tf.keras.layers.LSTM(layers[1], input_shape=(None, layers[0]),  return_sequences=True))
#model.add(Dropout(0.2))
model.add(tf.keras.layers.LSTM(layers[2],return_sequences=False))
#model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(layers[3], activation=tf.keras.activations.linear))

model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1, shuffle=False)

#Once the model is fit, we can estimate the performance of the model on the train and
# test datasets. This will give us a point of comparison for new models.

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Note that we invert the predictions before calculating error scores to ensure that performance is
#reported in the same units as the original data (thousands of passengers per month).
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore)) # 1-28, 5-25, 3-33, 10-34, Newshape: 5-26
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore)) # 1-51, 5-60, 3-70, 10-65, Newshape: 5-42

#Finally, we can generate predictions using the model for both the train and test dataset to get a
#visual indication of the skill of the model.
#Because of how the dataset was prepared, we must shift the predictions so that they align on the x-axis
#with the original dataset.

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)-1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), color = 'blue', label="Actual")
plt.plot(trainPredictPlot, color = 'red', label="Train_Forecast")
plt.plot(testPredictPlot, color = 'green', label="Test_Forecast")
plt.legend(loc="upper left")
plt.show()

del(data, dataset, train, trainPredict, trainPredictPlot, trainX, trainY, test, testPredict, testPredictPlot, testX, testY); gc.collect()

#%% LSTM using Keras - working with train mode only
#http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/#comment-409969
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Now, we can load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('./data/AirPassengers.csv')

#date time conversion
data['TravelDate'] =  pd.to_datetime(data['TravelDate'], format='%m/%d/%Y')
data['Passengers'] = data['Passengers'].astype(float)
data.dtypes # Notice the dtype=’datetime[ns]’

#Index: getting time to index
data.set_index('TravelDate', inplace=True)
data.head()

# Just to view
TS = data['Passengers']
TS.head(2)
TS.plot()

#Convert data into array that can be broken up into training "batches" that we will
#feed into our RNN model.
dataset_org = np.array(TS).astype('float32')

# Follwoing function will shift one element right and return difference only
#data_org = np.array([4, 3, 2])
def get_diff(data_org):
    data_org = np.array(data_org)
    data_diff = data_org[1:] - data_org[:-1]
    return(data_diff)
# get_diff

#data_diff = get_diff(data_org)
#data_org.size, data_diff.size

# It inverse the diff
#last_data_point_of_org = data_org[-1]; diff_pred = np.array([3, 1])
def inverse_diff(last_data_point_of_org, diff_pred):
    pred = np.append(np.ravel(last_data_point_of_org), np.ravel(np.array(diff_pred)))
    pred = np.cumsum(pred)
    pred = pred[1:]
    return(pred)

# inverse_diff(last_data_point_of_org, diff_pred)

# To make stationary, let us take diff. It will shorten the length by one(first element
# will be removed)
dataset = get_diff(dataset_org)
dataset_org.size, dataset.size

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1)) # The default activation function for LSTMs is the hyperbolic tangent (tanh), which outputs values between -1 and 1.
dataset = dataset.reshape(-1, 1) # Follwoing function need in this shape else deprecation warning
dataset = scaler.fit_transform(dataset)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, b_train_only = False):
    if b_train_only:
        dataX = []
        for i in range(len(dataset)-look_back+1): # i = 138
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
        return(np.array(dataX))
    else:
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back): # i = 138
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return(np.array(dataX), np.array(dataY))

# reshape into X=t and Y=t+1 using above function
look_back = 5
trainX, trainY = create_dataset(dataset, look_back)
trainX.shape, trainY.shape
#scaler.inverse_transform(trainX[0]), scaler.inverse_transform(trainY[0:1])
#scaler.inverse_transform(trainX[trainX.shape[0]-1]), scaler.inverse_transform(trainY[(trainY.shape[0]-1):trainY.shape[0]])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
trainX.shape

# create and fit the LSTM network
model = tf.keras.models.Sequential()
layers = [1, 100, 100, 100, 1] # [1, 50, 75, 100, 1]
model.add(tf.keras.layers.LSTM(layers[1], input_shape=(None, layers[0]),  return_sequences=True)) #  note that you only need to specify the input size on the first layer.
#model.add(Dropout(0.2))
model.add(tf.keras.layers.LSTM(layers[2],return_sequences=True)) # to stack recurrent layers, you must use return_sequences=True on any recurrent layer that feeds into another recurrent layer.
model.add(tf.keras.layers.LSTM(layers[3],return_sequences=False))
#model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(layers[4], activation=tf.keras.activations.linear))

model.compile(loss="mae", optimizer="adam") # rmsprop adam

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1, shuffle=False)

# Predict for next 12 steps.
pred_time_steps = 12; pred =  np.array([]) # np.array(range(pred_time_steps), dtype = np.float64)
for time_step in range(pred_time_steps): # time_step = 0
    # Store previous prediction
    #trainX_full = np.append(dataset, trainPredict[len(trainPredict)-1-time_step:len(trainPredict)])

    #Default
    trainX_full = dataset # Default. Last6:  [1.], [ 0.96911204],  [ 0.77992272], [ 0.6891892 ], [ 0.55212355], [ 0.63320458]

    if len(pred) == 0:
        trainX_full = dataset[-look_back:]
    else:
        trainX_full = np.append(dataset, pred)
        trainX_full = trainX_full[-look_back:]

    trainX_full = trainX_full.reshape(-1, 1) # Follwoing function need in this shape else deprecation warning

    # Create new train set including the last one
    trainX_full = create_dataset(trainX_full, look_back, b_train_only= True)

    # reshape input to be [samples, time steps, features]
    trainX_full = np.reshape(trainX_full, (trainX_full.shape[0], trainX_full.shape[1], 1))

    # make predictions
    trainPredict = model.predict(trainX_full)

    # Get the last value. Note: The first prediction is already in main dataset and
    #hence not adding at top of for loop
    pred = np.append(pred, trainPredict[len(trainPredict)-1])
    del(trainX_full)
    # for time_step in range(pred_time_steps):

# Inverse scalling
# It has both original and pred too
trainPredictPlot = np.append(dataset, pred)
trainPredictPlot = trainPredictPlot.reshape(-1, 1) # Follwoing function need in this shape else deprecation warning
trainPredictPlot = scaler.inverse_transform(trainPredictPlot)

# Bring back in pure numpy from share (-1,1) to (-1)
trainPredictPlot = np.array(np.ravel(trainPredictPlot))
trainPredictPlot[0:len(dataset)] = np.nan
trainPredictPlot[-pred_time_steps:] = inverse_diff(dataset_org[-1], trainPredictPlot[-pred_time_steps:])

# Check only
len(dataset_org), len(dataset), len(trainPredictPlot)

# plot baseline and predictions
plt.plot(dataset_org, color = 'blue', label="Actual")
plt.plot(trainPredictPlot, color = 'green', label="Forecast")
plt.legend(loc="upper left")
plt.show()

#CW: Do the self predict on train and clculate RMSE
#CW: How to increase accuracy - Epoch, Batch size, LSTM Layer, Neuron Gates, Dropout,

#%% Multivariate Time Series using LSTM
#https://archive.ics.uci.edu/ml/machine-learning-databases/00381/
import numpy as np
import pandas as pd
import math, os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc; gc.enable()

os.chdir("D:\\trainings\\tensorflow")

# fix random seed for reproducibility
seed = 123; np.random.seed(seed); tf.compat.v1.set_random_seed(seed)

#Now, we can load the data set and look at some initial rows and data types of the columns:
data = pd.read_csv('./data/PRSA_data_2010.1.1-2014.12.31.csv')

#Remove special char '.' from column name
data.columns = data.columns.str.replace('.', '')

#Few constants for easy reading
col_date = 'date'
col_label = 'pm25'

#date time conversion. Converting for each day only just to keep data short and simple
data[col_date] =  pd.to_datetime(data[['year', 'month', 'day']], format='%Y %m %d')

#Drop unwanted columns
data.drop(['No','year', 'month', 'day', 'hour','cbwd'], axis=1, inplace=True)

#See first 24 rows missing response value
#data.dropna(inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

#Aggregate or each day only just to keep data short and simple
data = data.groupby([col_date], as_index=False).mean()

col_features = np.setdiff1d(data.columns,np.array([col_date, col_label]))

data[np.append(np.array([col_label]), np.array(col_features))] = data[np.append(np.array([col_label]), np.array(col_features))].apply(lambda x: x.astype(np.float64))
data.dtypes # Notice the dtype=’datetime[ns]’

#First view
plt.plot(data[col_date], data[col_label])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1)) # The default activation function for LSTMs is the hyperbolic tangent (tanh), which outputs values between -1 and 1.
data[col_features] = scaler.fit_transform(data[col_features])
data.head(2)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX =[]
    dataY =[]
    for row_num in range(len(dataset)-look_back):
        one_dataset=[]
        for j in range(0,look_back):
            one_dataset.append(dataset[[(row_num+j)], 2:])
        
        dataX.append(one_dataset)
        dataY.append(dataset[row_num + look_back-1,1])
    
    return np.array(dataX, dtype = np.float64), np.array(dataY, dtype = np.float64)
#end

# reshape into X=t and Y=t+1 using above function
look_back = 5
trainX, trainY = create_dataset(data.values, look_back)

# reshape input to be [samples, time steps, features]
trainX = trainX.reshape(trainX.shape[0], look_back, len(col_features))
trainX.shape,trainY.shape

#Now let us see how these have been prepared
data.head(6)
trainX[0]
trainY[0]

data.tail(5)
trainX[-1]
trainY[-1]

# create and fit the LSTM network
model = tf.keras.models.Sequential()
layers = [1, 100, 100, 100, 1] # [1, 50, 75, 100, 1]
model.add(tf.keras.layers.LSTM(layers[1], input_shape=(trainX.shape[1],trainX.shape[2]),  return_sequences=True)) #  note that you only need to specify the input size on the first layer.
model.add(tf.keras.layers.LSTM(layers[2],return_sequences=True)) # to stack recurrent layers, you must use return_sequences=True on any recurrent layer that feeds into another recurrent layer.
model.add(tf.keras.layers.LSTM(layers[3],return_sequences=False))
model.add(tf.keras.layers.Dense(layers[4], activation=tf.keras.activations.linear))

model.summary()

model.compile(optimizer="adam", loss="mae")

model.fit(trainX, trainY, epochs=1000, batch_size=32, verbose=1, shuffle=False)

pred= np.ravel(model.predict(trainX))

np.max(trainY), np.min(trainY)
np.max(pred), np.min(pred)

# plot baseline and predictions
plt.plot(trainY, color = 'blue', label="Actual")
plt.plot(pred, color = 'green', label="Forecast")
plt.legend(loc="upper left")
plt.show()

del(data, col_date, col_label, col_features, scaler, look_back, trainX, trainY, model, layers, pred)
#%% CW: Practice GRU
#%% CW: Practice Bidirectional LSTM
#%% ppt: How to know models are good enough: Bias vs Variance
