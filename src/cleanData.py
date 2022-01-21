import pandas as pd

def commenceClean(dataframe):

  df = dataframe

  gender_map = {"Male": 1, "Female": 0}
  smoke_map = { "Yes": 1, "No": 0}
  diabetes_map = { "Normal": 0, "Pre-diabetes": 1, "Diabetes": 2 }
  ejection_map = { "Low": 0, "Normal": 1, "High": 2 }
  color_map = { "black": 0,"blue": 1, "green": 2,"red": 3, "white": 4, "yellow": 5}

  print('')
  print("### Cleaning up key feature: Survive...")
  df["Survive"] = df["Survive"].replace('No', 0)
  df["Survive"] = df["Survive"].replace('Yes', 1)
  df["Survive"] = df["Survive"].astype(int)
  print('[\u2713] Cleaned up key feature: survive')

  print('')
  print("### Ensuring ages are the correct inputs...")
  df["Age"] = abs(df["Age"])
  print('[\u2713] Checked ages')

  print('')
  print("### Mapping gender to categorical markers.")
  df["Gender"] = df["Gender"].map(gender_map)
  print('[\u2713] Mapped genders')

  print('')
  print("### Parsing BMI as a function of height and weight...")
  df["BMI"] = (df["Weight"] / (df["Height"] / 100 )**2)
  df = df.drop(columns=["Weight", "Height"])
  print('[\u2713] BMI computed')

  print('')
  print("### Cleaning up smoker details and mapping smokers to categories...")
  df["Smoke"] = df["Smoke"].replace("YES", "Yes")
  df["Smoke"] = df["Smoke"].replace("NO","No")
  df["Smoke"] = df["Smoke"].map(smoke_map)
  print('[\u2713] Smokers categorised.')

  print('')
  print("### Mapping diabetics to categorical markers...")
  df["Diabetes"] = df["Diabetes"].map(diabetes_map)
  print('[\u2713] Diabetics mapped')

  print('')
  print("### Cleaning ejection fraction and mapping to categorical markers...")
  df["Ejection Fraction"] = df["Ejection Fraction"].replace("L", "Low")
  df["Ejection Fraction"] = df["Ejection Fraction"].replace("N", "Normal")
  df["Ejection Fraction"] = df["Ejection Fraction"].replace("H", "High")
  df["Ejection Fraction"] = df["Ejection Fraction"].map(ejection_map)
  print('[\u2713] EF done.')

  print('')
  print("### Grouping and mapping colors...")
  df["Favorite color"] = df["Favorite color"].map(color_map)
  print('[\u2713] Colors grouped')

  return df