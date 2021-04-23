import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# we could try linear regression
from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("Vaccinations") \
    .getOrCreate()


url = "https://data.chhs.ca.gov/dataset/e283ee5a-cf18-4f20-a92c-ee94a2866ccd/resource/130d7ba2-b6eb-438d-a412-741bde207e1c/download/covid19vaccinesbycounty.csv"
from pyspark import SparkFiles
spark.sparkContext.addFile(url)

df = spark.read.csv("file://"+SparkFiles.get("covid19vaccinesbycounty.csv"), header=True, inferSchema= True)
df.printSchema()


## ONLY INCLUDES ROWS WHERE california_flag is empty - we could instead get a vaccination rate per county
print("all rows", df.count())
df = df.filter(df.california_flag.isNull())
print("only california", df.count())


## CONVERT TO DATE TYPE
df = df.select('*', df.administered_date.cast('date').alias('date'))

import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import autocorrelation_plot

data = np.array(df.select("date", "fully_vaccinated").sort("date").collect())
x = [it[0] for it in data]
y = [it[1] for it in data]
plt.plot(x,y)
plt.savefig("ca_vaccination_rate_over_time.png")
plt.show()


# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# we could use the above tutorial to predict vaccination rates
# autocorrelation_plot([x, y])