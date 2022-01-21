import pandas as pd
import numpy as np

def commenceCat(cleanedDf):

  df = cleanedDf

  print('')
  print("### Splitting BMI into categories...")
  df["BMI"] = pd.qcut(df["BMI"],4, labels=[0,1,2,3])
  print('[\u2713] BMI categorized.')
  print('')

  print("### Splitting ages into categories...")
  df["Age"] = pd.qcut(df["Age"], 4, labels=[0,1,2,3])
  print('[\u2713] Ages categorized.')
  print('')

  print("### Splitting sodium into categories...")
  df["Sodium"] = pd.qcut(df["Sodium"], 5, labels=[0,1,2,3,4])
  print('[\u2713] Sodium categorized.')
  print('')

  print("### Replacing missing values for creatinine...")
  mean = df["Creatinine"].mean()
  std = df["Creatinine"].std()
  null_values = df["Creatinine"].isnull().sum()
  random_creatinine = np.random.randint( mean-std, mean+std, size = null_values)
  creatinine_copy = df["Creatinine"].copy()
  creatinine_copy[np.isnan(creatinine_copy)] = random_creatinine
  df["Creatinine"] = creatinine_copy
  df["Creatinine"] = pd.qcut(df["Creatinine"], 4, labels=[0,1,2,3])
  print('[\u2713] Creatinine categorized.')
  print('')

  print("### Splitting platelets into categories...")
  df["Platelets"] = pd.qcut(df["Platelets"], 4, labels=[0,1,2,3])
  print('[\u2713] Platelets categorized.')
  print('')

  print("### Splitting creatine phosphokinase into categories...")
  df["Creatine phosphokinase"] = pd.qcut(df["Creatine phosphokinase"], 5, labels=[0,1,2,3,4])
  print('[\u2713] Creatine phosphokinase categorized.')
  print('')

  print("### Splitting blood pressure into categories...")
  df["Blood Pressure"] = pd.qcut(df["Blood Pressure"], 4, labels=[0,1,2,3])
  print('[\u2713] Blood Pressure categorized.')
  print('')

  print("### Splitting Hemoglobin into categories...")
  df["Hemoglobin"] = pd.qcut(df["Hemoglobin"], 4, labels=[0,1,2,3])
  print('[\u2713] Hemoglobin categorized.')
  print('')

  df = df.drop(columns = ["ID"])

  return df