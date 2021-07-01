# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:35:06 2021

@author: jamalm
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Apply the default theme
sns.set_theme()
sns.set(font_scale=1.5)
plt.style.context("seaborn-whitegrid")
sns.set_style('whitegrid')

class MachineLearningModel:
  def __init__(self,model,name,testDataFrame, X_test, OutputDataFrame = None):
    self.model = model
    self.name = name
    self.testDataFrame = testDataFrame
    self.X_test = X_test

    # Values of the predictions and calculations are stored in the test dataframe by default unless an alternate dataframe is specified 
    if OutputDataFrame is not None:
      self.OutputDataFrame = OutputDataFrame
    else:
      self.OutputDataFrame = self.testDataFrame

  def predict(self):    
    # Function predicts the Health Index values based on the trained machine learning model.
    Prediction = self.model.predict(self.X_test)

    # The predicted H.I. values are appended into the dataframe 
    self.OutputDataFrame[f'Pred_H.I._{self.name}'] = Prediction

    # Calculating the predicted final cycle of the engine units      
    self.OutputDataFrame[f'PredFinalCycle_{self.name}'] = self.OutputDataFrame['cycles'] / (1 - self.OutputDataFrame[f'Pred_H.I._{self.name}'])

  def PlotPredFinalCycle(self,displayUnit=1, display_mean = True):
    # Returning the average final cycle value of a particular engine unit
    df = self.OutputDataFrame[f'PredFinalCycle_{self.name}']
    meanValue = df[(self.OutputDataFrame['unit'] == displayUnit)][-10:].mean()
    print(f'Mean value of last 10 cycles of engine unit {displayUnit}: {meanValue}') 

    # plotting the predicted final cycles
    plt.figure(figsize = (18, 10))
    y_axis = df[(self.OutputDataFrame['unit'] == displayUnit)]
    x_axis = self.OutputDataFrame['cycles'][self.OutputDataFrame['unit'] == displayUnit]

    plt.plot(x_axis, y_axis, linewidth=2.5, label='Predicted Final Cycle')
    plt.xlabel('Cycles')
    plt.ylabel('Predicted Final Cycle')
    if display_mean == True:
      plt.axhline(meanValue, color="red", linestyle='dashed', linewidth=1.6, label='Calculated mean')

    # set title and axis labels
    errorDist = plt.title(f'Predicted Final Cycle of engine unit {displayUnit}', x=0.5, y=1.05, ha='center', fontsize='xx-large')
    plt.setp(errorDist, color='black')
    plt.legend(loc='best')

  def DataFrame(self, displayUnit=None, remove_sensor_data=False):
    df = self.OutputDataFrame
    if displayUnit:
      df = df[df['unit'] == displayUnit]

    # function returns dataframe
    droppedColumns = ['Op1', 'Op2', 'S2', 'S3', 'S4', 'S7', 'S8', 'S11', 'S12',
          'S13', 'S14', 'S15', 'S17', 'S20', 'S21']
    if remove_sensor_data == True:
      df.drop(droppedColumns, axis=1, errors='ignore', inplace=True)
    return df

  def CalcPredFinalCycle(self):
    df = self.OutputDataFrame[f'PredFinalCycle_{self.name}']

    # Returns a uniform PredFinalCycle based on the calculated average of the last 10 cycles 
    for i in range(self.OutputDataFrame['unit'].min(), self.OutputDataFrame['unit'].max() + 1):
        df[(self.OutputDataFrame['unit'] == i)] = df[(self.OutputDataFrame['unit'] == i)][-10:].mean()
  
  def CalcPredRUL(self):
    self.OutputDataFrame[[f'Pred_RUL_{self.name}']] = self.OutputDataFrame[f'PredFinalCycle_{self.name}'] * self.OutputDataFrame[f'Pred_H.I._{self.name}']

    # Appending the RUL values of each engine unit into an array
    df =  self.OutputDataFrame[f'Pred_RUL_{self.name}']
    pred_final = []
    for i in range(1,101):
      pred_final.append(df[(self.OutputDataFrame['unit'] == i)].min())

    self.predicted = pred_final
    return pred_final  

  def PlotRUL(self, displayUnit = None, line_width=2.5, show_final_RUL=False, show_best_fit=False):
    # Visualizing the predicted cycles of the engine units in the test dataset.
    # x-axis represents the cycles
    engine_unit = ''
    if displayUnit is None:
      y_axis =  self.OutputDataFrame[f'Pred_RUL_{self.name}']
      x_axis = range(len(self.OutputDataFrame['cycles']))
      plt.figure(figsize = (18, 10)) 
    else:
      y_axis =  self.OutputDataFrame[f'Pred_RUL_{self.name}'][self.OutputDataFrame['unit'] == displayUnit]
      x_axis = self.OutputDataFrame['cycles'][self.OutputDataFrame['unit'] == displayUnit]
      engine_unit = f'Engine {displayUnit}, '
      plt.figure(figsize = (20, 5)) 
   
    if show_final_RUL:
      min_RUL = y_axis[(self.OutputDataFrame['unit'] == displayUnit)].min()
      plt.axhline(min_RUL, color="red", linestyle='dashed', linewidth=1.6, label='Final Cycle')
      print(f"RUL of engine unit {displayUnit} is {min_RUL}")
      

    plt.plot(x_axis, y_axis, linewidth=line_width, label='Predicted RUL')
    plt.xlabel('Cycles')
    plt.ylabel('Predicted RUL')
    plt.title(f'Predicted Remaining Useful Life; {engine_unit}{self.name}', x=0.5, y=1.05, ha='center', fontsize='x-large')
    
    if show_best_fit:
      bf_x_axis = x_axis.values.reshape(-1,1)
      RUL_lreg = LinearRegression()
      RUL_lreg.fit(bf_x_axis,y_axis)
      best_fit = RUL_lreg.predict(bf_x_axis)
      plt.plot(x_axis, best_fit, linewidth=.75*line_width, label='Best fit', linestyle='dashed')
    
    plt.legend(loc='best')

  def RUL_bestFit(self):
 
    for i in range(self.OutputDataFrame['unit'].min(), self.OutputDataFrame['unit'].max() + 1):
      x_axis = self.OutputDataFrame['cycles'][self.OutputDataFrame['unit'] == i]
      y_axis =  self.OutputDataFrame[f'Pred_RUL_{self.name}'][self.OutputDataFrame['unit'] == i]

      bf_x_axis = x_axis.values.reshape(-1,1)
      RUL_lreg = LinearRegression()
      RUL_lreg.fit(bf_x_axis,y_axis)
      self.OutputDataFrame[f'Pred_RUL_{self.name}'][self.OutputDataFrame['unit'] == i] = RUL_lreg.predict(bf_x_axis)

    df =  self.OutputDataFrame[f'Pred_RUL_{self.name}']
    pred_final = []
    for i in range(1,101):
      pred_final.append(df[(self.OutputDataFrame['unit'] == i)].min())

    self.predicted = pred_final
    return pred_final  

  def DisplayEngineRUL(self, displayUnit=1):
    # Returning the Remaining useful life of an engine unit of the test dataset.
    # It is found by returning the minumum cycle to failure value of each engine unit
    df =  self.OutputDataFrame[f'Pred_RUL_{self.name}'] 
    value = df[(self.OutputDataFrame['unit'] == displayUnit)].min()
    print(f'Predicted RUL of unit {displayUnit}: {value}')

  def GetRMSE(self,actual,predicted=None):
    if predicted is None:
      predicted = self.predicted
    
    # Calculating the Root Mean Square Error of the actual and predicted values
    RMSE = mean_squared_error(actual, predicted, squared=False)
    return RMSE
