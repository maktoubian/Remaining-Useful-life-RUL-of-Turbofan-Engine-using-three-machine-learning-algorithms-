# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:20:24 2021

@author: jamalm
"""
import matplotlib.pyplot as plt
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


#Importing both test and train datasets
df_test_FD001 = pd.read_csv('...\\Datasets\\FD001\\test_FD001.txt', sep=' ', header=None)
df_train_FD001 = pd.read_csv('...\\Datasets\\FD001\\train_FD001.txt', sep=' ', header=None)


# dropping NAN values
df_test_FD001 = df_test_FD001.dropna(axis=1, how='all')
df_train_FD001 = df_train_FD001.dropna(axis=1, how='all')

# Naming the columns
df_test_FD001.columns = ["unit", "cycles", "Op1",
                      "Op2", "Op3", "S1", "S2",
                      "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
                      "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21"]

df_train_FD001.columns = ["unit", "cycles", "Op1",
                      "Op2", "Op3", "S1", "S2",
                      "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
                      "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21"]

# data preprocessing; removing unnecessary data
df_test_FD001.drop(['Op3','S1', 'S5', 'S6', 'S9', 'S16', 'S10', 'S18', 'S19'], axis=1, inplace=True)
df_train_FD001.drop(['Op3','S1', 'S5', 'S6', 'S9', 'S16', 'S10', 'S18', 'S19'], axis=1, inplace=True)

# test dataset begins with unit 101 to differentiate from the train dataset
df_test_FD001['unit'] += 100

# Combine the two datasets into a single dataframe
df_combined = pd.concat([df_train_FD001, df_test_FD001]) 

# Perform scalling on the the combined dataset so that their scale is the same
scaler = MinMaxScaler()
df_combined.iloc[:,2:18] = scaler.fit_transform(df_combined.iloc[:,2:18])

# Split them again after performing scalling
df_train_FD001 = df_combined[(df_combined.unit <= 100)]
df_test_FD001 = df_combined[(df_combined.unit >= 101)]

# After separation, the first unit of the test dataset is reset to 0 again
df_test_FD001['unit'] -= 100



#############################################################################
#Calculate the remaining Useful life of engin units
#############################################################################

# Finding the maximum cycle of an engine unit which is used to find its Remaining Useful Life (RUL)
df_train_FD001 = pd.merge(df_train_FD001, df_train_FD001.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
df_train_FD001.rename(columns={"cycles_x": "cycles", "cycles_y": "Final Cycle"}, inplace=True)

#  Remaining Useful Life of an engine unit at a cycle is calculated by subtracting its final cycle 
#  (after which the engine stops operating or also the maximum/total number of cycles), to the current cycle
# Add a new colomn named RUL
df_train_FD001['RUL'] = df_train_FD001['Final Cycle'] - df_train_FD001['cycles']

# Defining the Health Index, where value of 1 denotes healthy engine and 0 denotes failure
def HealthIndex(dataFrame,q):
    return(dataFrame.RUL[q]-dataFrame.RUL.min()) / (dataFrame.RUL.max()-dataFrame.RUL.min())

healthIndex_q = []
healthIndex = []

for i in range(df_train_FD001['unit'].min(), df_train_FD001['unit'].max() + 1):
    dataFrame = df_train_FD001[df_train_FD001.unit == i]
    dataFrame = dataFrame.reset_index(drop = True)
    for q in range(len(dataFrame)):
        healthIndex_q = HealthIndex(dataFrame, q)
        healthIndex.append(healthIndex_q)

df_train_FD001['Health Index'] = healthIndex


# Defining train values that will be used to train the machine learning model
X_train = df_train_FD001[['cycles', 'Op1', 'Op2', 'S2', 'S3', 'S4', 'S7', 'S8', 'S11', 'S12',
          'S13', 'S14', 'S15', 'S17', 'S20', 'S21']].values
y_train = df_train_FD001[['Health Index']].values.ravel()

# Defining test values that will be used to perform prediction based on the trained model
X_test = df_test_FD001[['cycles', 'Op1', 'Op2', 'S2', 'S3', 'S4', 'S7', 'S8', 'S11', 'S12',
          'S13', 'S14', 'S15', 'S17', 'S20', 'S21']].values

###################################################
# Linear Regrassion Model
from sklearn.linear_model import LinearRegression
lreg_model = LinearRegression()
lreg_model.fit(X_train,y_train)

###################################################
#KNN Regressor Model
from sklearn.neighbors import KNeighborsRegressor
knr_model = KNeighborsRegressor(n_neighbors=12)
knr_model.fit(X_train, y_train)

###################################################
#Neural Network Model
# Importing keras deep learning API into colab notebook
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor

# Defining Neural Network model
nn_model = Sequential()
nn_model.add(Dense(50, input_dim=16, kernel_initializer='normal', activation='relu'))
nn_model.add(Dense(1, activation='relu'))

# Compiling model
nn_model.compile(loss='mean_squared_error', optimizer='adam')

# Training the Neural Network model
nn_model.fit(X_train, y_train, epochs = 55, shuffle=True, verbose = 0)

#The best Neural Network model as a result of multiple runs was saved and can be uploaded again 
#so that the most accurate model can be retrieved again for an accurate prediction.
import keras
# from tensorflow import keras
nn_model = keras.models.load_model('...\\Datasets\\BestModel')

Results_nn = df_test_FD001[['unit', 'cycles']]
Results_knr = df_test_FD001[['unit', 'cycles']]
Results_lreg = df_test_FD001[['unit', 'cycles']]

###########################################
#import the MachineLearningModel class in order to reduce the code duplication
from MachineLearningModel import MachineLearningModel
NeuralNetwork = MachineLearningModel(nn_model, 'nn', df_test_FD001, X_test, Results_nn)
NeuralNetwork.predict()
NeuralNetwork.CalcPredFinalCycle()
RUL_nn = NeuralNetwork.CalcPredRUL()
NeuralNetwork.DataFrame()


###########################################
#K Nearest Neighbour Regressor prediction
KNRegressor = MachineLearningModel(knr_model, 'knr', df_test_FD001, X_test, Results_knr)
KNRegressor.predict()
KNRegressor.CalcPredFinalCycle()
RUL_knr = KNRegressor.CalcPredRUL()
KNRegressor.DataFrame()

###########################################
# Linear regression model prediction
LinearRegression = MachineLearningModel(lreg_model, 'lreg', df_test_FD001, X_test, Results_lreg)
LinearRegression.predict()
LinearRegression.CalcPredFinalCycle()
RUL_lreg = LinearRegression.CalcPredRUL()
LinearRegression.DataFrame()

#Residuals: Machine learning/statistic model shows the differences between observed and predicted values of data
############################################
FinalResult = pd.read_csv('...\\Datasets\\FD001\\RUL_FD001.txt', sep=' ', header=None)
FinalResult = FinalResult.dropna(axis=1, how='all')
FinalResult.columns = ["Actual RUL"]

unit = []
for i in range(1,101):
  unit.append(i)
FinalResult['Unit'] = unit

names = ['lreg', 'knr', 'nn']
models = [RUL_lreg, RUL_knr, RUL_nn]

for i in range(len(names)):
  predRUL_column_name = f'Pred_RUL_{names[i]}'
  diff_column_name = f'diff_{names[i]}'
  FinalResult[predRUL_column_name] = models[i]
  FinalResult[diff_column_name] = FinalResult['Actual RUL'] - FinalResult[predRUL_column_name]



# Kernel Density Estimation (KDE) Plotting: Error distribution can help to Compare accuracy between the machine learning models
###################################################
names = ['lreg', 'knr', 'nn']
lables = ['Linear Regression','K Nearest Neighbour','Neural Network']

plt.figure(figsize=(18,10))
sns.set_style('darkgrid')

for i in range(len(names)):
  FinalResult[f'diff_{names[i]}'].plot(label = lables[i], kind = 'kde', linewidth=4.5, legend = True)
  sns.kdeplot(FinalResult[f'diff_{names[i]}'], shade=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,)
plt.axvline(0, color="black", linestyle='dashed', linewidth=3)

# set title and axis labels
errorDist = plt.title('Error Distribution', x=0.5, y=1.05, ha='center', fontsize='30')
plt.setp(errorDist, color = 'Black')
plt.xlabel('Error')
plt.show();



#Root Mean Square Error (RMSE): square root of the average of the squared differences between
# the actual and predicted values. 
RMSE_nn = NeuralNetwork.GetRMSE(FinalResult['Actual RUL'])
RMSE_knr = KNRegressor.GetRMSE(FinalResult['Actual RUL'])
RMSE_lreg = LinearRegression.GetRMSE(FinalResult['Actual RUL'])
#print the RMSE results for each model
print('Neural Network RMSE:', RMSE_nn)
print('K Nearest Neighbour RMSE:', RMSE_knr)
print('Linear Regression RMSE:', RMSE_lreg)
