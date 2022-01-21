from sqlite3 import connect
import sqlalchemy as sql
import pandas as pd

def connectToDB(connectionString, echo):
  db = sql.create_engine(connectionString, echo=echo)
  engine = db.connect()
  query_all = "SELECT * FROM survive"
  df = pd.read_sql(query_all, engine)

  return df

