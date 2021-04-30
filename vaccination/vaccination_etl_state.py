import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# we could try linear regression
from pyspark.ml.regression import LinearRegression

from pyspark.ml.feature import  VectorIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("Vaccinations") \
    .getOrCreate()


url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv"
from pyspark import SparkFiles
spark.sparkContext.addFile(url)

df = spark.read.csv("file://"+SparkFiles.get("us_state_vaccinations.csv"), header=True, inferSchema= True).select("date", "location", "daily_vaccinations")

# ONLY INCLUDES ROWS WHERE california_flag is empty - we could instead get a vaccination rate per county
print("all rows", df.count())
df = df.na.drop()
print("only non null", df.count())

## CONVERT TO DATE TYPE
df = df.select('*', unix_timestamp(df.date.cast('date')).alias('time'))

states = df.rdd.map(lambda x : x.location).distinct().collect()

import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import autocorrelation_plot

from pyspark.ml.regression import LinearRegression
lr = LinearRegression()

for state in ["United States"]:
  print("processing state", state)
  data = df.select("time", "daily_vaccinations").filter(df.location == state).sort("time")
  d = np.array(data.collect())
  x = [it[0] for it in d]
  y = [it[1] for it in d]

  last_time = x[-1]

  test_x = [last_time+(idx*86400) for idx in range(1,31)] #get epoch time for the next 30 days

  train = data.rdd.map(lambda x:(Vectors.dense(x[0]), x[1])).toDF(["features", "label"])
  train.show()

  linear_model = lr.fit(train)

  print("Coefficients: " + str(linear_model.coefficients))
  print("\nIntercept: " + str(linear_model.intercept))

  all_X = np.concatenate([x,test_x])

  t = [(Vectors.dense(i), 0) for i in test_x]

  inp = spark.sparkContext.parallelize(t).toDF(["features", "label"])

  test03 = linear_model.transform(inp)
  # test03.show(truncate=False)

  test03.show()

  preds = np.array(test03.select("features", "prediction").collect())
  pred_x = [it[0].toArray()[0] for it in preds]
  pred_y = [it[1] for it in preds]

  all_Y = np.concatenate([y, pred_y])

  numRows = all_Y.shape[0]

  plt.plot(all_X, all_Y)
  plt.plot(x,y)
  plt.savefig("plots/"+state+"_vaccination_rate_over_time.png")
  plt.clf()

  locs = np.full(numRows, state)
  print(locs.shape)
  print(all_X.shape)
  print(all_Y.shape)
  print(np.dstack((locs, all_X,all_Y)).shape)
  newfile = np.savetxt("predictions.csv", np.dstack((locs, all_X,all_Y))[0],"%s,%s,%s",header="location,epoch,vaccination_rate")
  print(newfile)
