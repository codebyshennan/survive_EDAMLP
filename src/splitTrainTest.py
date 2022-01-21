from sklearn.model_selection import train_test_split

def split(categorizedDf):
  df = categorizedDf
  seed = 7
  train, test = train_test_split(df, test_size=0.2, random_state=seed)

  return train, test