import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)

#Load the sequence data from csv
df = pd.read_csv('/home/uduwela96/flask-black-dashboard/apps/model/AEP_hourly.csv')
requests = pd.DataFrame(data=df,columns=['AEP_MW'])
#if any missing value fill it by previous value and convert all requests into integer type
requests.ffill(inplace=True)
requests["AEP_MW"]=requests["AEP_MW"].astype(float).astype(int)
#Review loaded data
#print(requests.dtypes)
#requests.head()
#print(requests)
##############################################################################################################
dataset = df
dataset["Month"] = pd.to_datetime(df["Datetime"]).dt.month
dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
dataset["Week"] = pd.to_datetime(df["Datetime"]).dt.isocalendar().week
dataset["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
dataset = df.set_index("Datetime")
dataset.index = pd.to_datetime(dataset.index)

##################################################################################################################
from sklearn.preprocessing import StandardScaler

#scale the data
#print("Request Range before scaling:",min(requests.AEP_MW),max(requests.AEP_MW))
scaler = StandardScaler()
scaled_requests = scaler.fit_transform(requests)
#print("Request Range after scaling:",min(scaled_requests),max(scaled_requests))

#Traing data has to be sequential
train_size =80410

#Number of samples to lookback for each sample
#720 default
lookback =720

#sperate training and test data
train_requests = scaled_requests[0:train_size,:]

#Add an additional week for lookback
test_requests = scaled_requests[train_size-lookback:,:]

#print("\n Shaped of Train ,Test :", train_requests.shape ,test_requests.shape)
##########################################################################################################
#Build a LSTM model with Keras
##########################################################################################################
#pepare RNN Dataset
def create_rnn_dataset(data, lookback=1):
  
  data_x,data_y = [],[]
  for i in range(len(data)- lookback -1):
    a = data[i:(i + lookback),0]
    data_x.append(a)
    data_y.append(data[i + lookback,0])
  return np.array(data_x),np.array(data_y)

#create x and y for training
train_req_x , train_req_y = create_rnn_dataset(train_requests , lookback)

#Reshape for use with LSTM
train_req_x = np.reshape(train_req_x,(train_req_x.shape[0],1,train_req_x.shape[1]))

#print("shapes of x,y:",train_req_x.shape , train_req_y.shape)
########################################################################################################
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
from keras.layers import Dropout, RepeatVector, TimeDistributed

tf.random.set_seed(21)
ts_model =  Sequential()
#Add LSTM
ts_model.add(LSTM(128,input_shape=(1,lookback)))
ts_model.add(Dense(1))
ts_model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mse"])
ts_model.summary()
history = ts_model.fit(train_req_x, train_req_y, epochs=5,validation_split=0.1, batch_size=1000,verbose=1)

###########################################################################################################
#Plot Training - Validation loss
###########################################################################################################
#plt.plot(history.history['loss'], label='Training loss')
#plt.plot(history.history['val_loss'], label='Validation loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend();
############################################################################################################
#Test the Model
#############################################################################################################
#Preprocess
test_req_x , test_req_y =create_rnn_dataset(test_requests , lookback)
test_req_x = np.reshape(test_req_x,(test_req_x.shape[0],1,test_req_x.shape[1]))
ts_model.evaluate(test_req_x , test_req_y, verbose=1)

#predict for the training dataset
predict_on_train = ts_model.predict(train_req_x)
#Prdeict on the test dataset
predict_on_test = ts_model.predict(test_req_x)

#train_mae_loss = np.mean(np.abs(predict_on_train - train_req_x), axis=1)

predict_on_train = scaler.inverse_transform(predict_on_train)
predict_on_test = scaler.inverse_transform(predict_on_test)

##############################################################################################################
#accuracy score
############################################################################################################
from sklearn import metrics
import os
score = np.sqrt(metrics.mean_squared_error(predict_on_test,test_req_y))
print(f'After training the score is:{score}')
###############################################################################################################
os.chdir(r'/home/uduwela96/flask-black-dashboard/apps/model/modelsave')
os.getcwd()
ts_model.save(os.path.join(os.getcwd(),"lstm_model.h5"))