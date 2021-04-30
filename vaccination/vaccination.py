import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark import SparkFiles
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("Vaccinations") \
    .getOrCreate()

# LOADING DATA
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv"
spark.sparkContext.addFile(url)
df = spark.read.csv("file://"+SparkFiles.get("us_state_vaccinations.csv"), header=True, inferSchema= True).select("date", "location", "daily_vaccinations")

# ONLY INCLUDES ROWS WHERE date, location, and daily_vaccinations are present
print("all rows", df.count())
df = df.na.drop()
print("only non null", df.count())

# CONVERT TO DATE TYPE
df = df.select('*', unix_timestamp(df.date.cast('date')).alias('time'))

# GET LIST OF ALL STATES
states = df.rdd.map(lambda x : x.location).distinct().collect()

# SAVE LIST OF STATES
statesfilepath = "states.csv"
newfile = np.savetxt(statesfilepath, states,"%s",header="states")

MODEL_TYPE = "LinearRegression"

for state in states: #["United States"]:
  print("PROCESSING STATE: ", state)

  # LINEAR REGRESSION
  lr = LinearRegression()

  # ONLY USE CURRENT STATE
  data = df.select("time", "daily_vaccinations").filter(df.location == state)
  dataArr = np.array(data.collect())
  x = [it[0] for it in dataArr]
  y = [it[1] for it in dataArr]

  # CALCULATE EPOCH TIME (X COORDINATE / INPUT) FOR THE NEXT 30 DAYS
  last_time = x[-1]
  test_x = [last_time+(idx*86400) for idx in range(1,31)] #get epoch time for the next 30 days

  # CREATE TRAINING SET
  train = data.rdd.map(lambda x:(Vectors.dense(x[0]), x[1])).toDF(["features", "label"])

  # TRAIN
  linear_model = lr.fit(train)

  # print("Coefficients: " + str(linear_model.coefficients))
  # print("\nIntercept: " + str(linear_model.intercept))

  # CREATE TEST DATA
  all_X = np.concatenate([x,test_x])
  test_arr = [(Vectors.dense(i), 0) for i in test_x]
  testData = spark.sparkContext.parallelize(test_arr).toDF(["features", "label"])

  # GET PREDICTIONS
  output = linear_model.transform(testData)
  # output.show()
  preds = np.array(output.select("features", "prediction").collect(), dtype='object')
  pred_y = [it[1] for it in preds]

  # ALL OUTPUTS FROM 1ST VACCINATION - 30 DAYS FROM NOW, PLOTTED
  all_Y = np.concatenate([y, pred_y])
  numRows = all_Y.shape[0]
  plt.plot(all_X, all_Y) #THIS WILL BE SAVED IN THE CSV
  plt.plot(x,y) #HISTORICAL DATA
  plt.savefig("plots/"+state+"_vaccination_rate_over_time.png")
  plt.clf()

  # SAVE TO CSV
  locs = np.full(numRows, state)
  filepath = "predictions/" + state + "_" + MODEL_TYPE + "_predictions.csv"
  newfile = np.savetxt(filepath, np.dstack((locs, all_X,all_Y))[0],"%s,%s,%s",header="location,epoch,vaccination_rate")
