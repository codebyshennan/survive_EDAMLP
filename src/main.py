import parseData
import splitTrainTest
import cleanData
import categorizeData
import trainModels
import argparse
import sys
import os

import datetime
import pandas as pd
import csv

tasktimes = {}
process = "start"

parser = argparse.ArgumentParser(description = "This script parses a database of patients suffering from coronary artery disease and trains it against various machine learning models to predict survival rates.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model', metavar='ML Model', required = True,
                    help='''
                    'all': run input against all ML models
                    'rf': Random Forests
                    'sgd': Stochastic Gradient Descent
                    'logreg': Logistic Regression
                    'knn': K-Nearest Neighbours
                    'naive': Naive Bayes
                    'svc': Support Vector Classification
                    'dt': Decision Trees
                    ''')
args = parser.parse_args()

### READ DATA ###
if process == "start":
  print('')
  print('')
  print(" > Reading into database...")
  start_time = datetime.datetime.now()
  connection_string = "sqlite:///data/survive.db"
  df = parseData.connectToDB(connection_string, True)
  tasktimes["Train-Test Split"] = (datetime.datetime.now() - start_time).total_seconds()
  process = "clean"

### CLEAN DATA ###
if process == "clean":
  print('')
  print('')
  print(" > Commencing data cleaning...")
  start_time = datetime.datetime.now()
  cleaned_df = cleanData.commenceClean(df)
  tasktimes["Clean Data"] = (datetime.datetime.now() - start_time).total_seconds()
  process = "categorize"

### CATEGORIZE AND GENERATE FEATURES ###
if process == "categorize":
  print('')
  print('')
  print(" > Commencing feature categorization and generation...")
  start_time = datetime.datetime.now()
  categorized_df = categorizeData.commenceCat(cleaned_df)
  tasktimes["Categorize Data"] = (datetime.datetime.now() - start_time).total_seconds()
  process = "split"

### TRAIN TEST SPLIT ###
if process == "split":
  print('')
  print('')
  print(" > Commencing train-test split...")
  start_time = datetime.datetime.now()
  train, test  = splitTrainTest.split(categorized_df)

  # save as CSV for future reuse after cleaning
  train.to_csv('./data/train_file.csv', index=False)
  print('[\u2713] Saved training file to data')

  test.to_csv('./data/test_file.csv', index=False)
  print('[\u2713] Saved testing file to data')
  print('')

  tasktimes["Train-Test Split"] = (datetime.datetime.now() - start_time).total_seconds()
  process = "predict"

### TRAIN & PREDICT ### 
if process == "predict":
  print('')
  print('')
  print(" > Commencing training and prediction...")
  start_time = datetime.datetime.now()

  X_train = pd.read_csv('./data/train_file.csv')
  X_test = pd.read_csv('./data/test_file.csv')

  print('[\u2713] Files parsed')

  Y_train = X_train["Survive"]
  X_train = X_train.drop("Survive", axis=1)
  X_test = X_test.drop("Survive", axis=1).copy()
  
  accuracy, prediction = trainModels.trainAndPredict(X_train, X_test, Y_train, args.model)

  acc_results = pd.DataFrame({
    'Model': accuracy.keys(),
    'Accuracy': accuracy.values()
  })

  acc_results = acc_results.sort_values(by="Accuracy", ascending = False)
  acc_results = acc_results.set_index("Accuracy")

  print('')
  print("Accuracy: ")
  print(acc_results)
  print('')

  pred_results = pd.DataFrame({
    'Model': prediction.keys(),
    'Prediction': prediction.values()
  })

  pred_results = pred_results.set_index("Prediction")

  print('')
  print("Prediction: ")
  print(pred_results)
  print('')

  tasktimes["Train and Predict"] = (datetime.datetime.now() - start_time).total_seconds()

totalTime = sum(tasktimes.values())
runTime_df = pd.DataFrame()
runTime_df["Task"] = tasktimes.keys()

if totalTime != 0:
  runTime_df["Percent of Pipeline"] = [ i / totalTime * 100 for i in tasktimes.values()]

  print('')
  print('')
  print("##########")
  print(runTime_df)
  print("##########")
  print('')
  print('[\u2713] Process completed')