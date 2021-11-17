import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark
from pyspark.ml.linalg import Vectors
import numpy as np
from sklearn.linear_model import LinearRegression
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt 
import time
from pandas import Series,DataFrame
import pandas as pd
import re
from collections import Counter
from sklearn.linear_model import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC


# building functions

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    
def correctRows(p):
    if isfloat(p[3]) and isfloat(p[4]) and isfloat(p[6]) and isfloat(p[7]) and isfloat(p[9]):
        return p
    
def to_list(a):
    return [a]

def addToList(x, y):
    x.append(y)
    return x

def extend(x,y):
    x.extend(y)
    return x

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

# load data set
lines2 = sc.textFile("Google-Playstore.csv")


print("##### Finding Simple Linear Regression Equation #####")

# data pre-processing
correctLine = lines2.map(lambda x: x.split(','))
cleaned = correctLine.filter(correctRows)

max_install = cleaned.map(lambda p: (float(p[7])))
rating = cleaned.map(lambda p: (float(p[3])))

# apply linear regression
x = np.array(max_install.collect())
y = np.array(rating.collect())

X = np.stack([x], axis = 1)

start2 = time.time()

rating = cleaned.map(lambda p: (p[0], p[3]))
rating_count = cleaned.map(lambda p: (p[0], p[4]))
min_install = cleaned.map(lambda p: (p[0], p[6]))
max_install = cleaned.map(lambda p: (p[0], p[7]))
price = cleaned.map(lambda p: (p[0], p[9]))

rating = rating.combineByKey(to_list, addToList, extend)
rating = rating.collect()
rating_count = rating_count.combineByKey(to_list, addToList, extend)
rating_count = rating_count.collect()
min_install = min_install.combineByKey(to_list, addToList, extend)
min_install = min_install.collect()
max_install = max_install.combineByKey(to_list, addToList, extend)
max_install = max_install.collect()
price = price.combineByKey(to_list, addToList, extend)
price = price.collect()


ratingKey = []
ratingValue = []
for i in range(len(rating)):
    ratingKey.append(rating[i][0])

    rate = 0
    total = 0
    for j in [float(i) for i in rating[i][1]]:
        rate += j
        total += 1

    ratingValue.append(rate/total)


ratingCountKey = []
ratingCountValue = []
for i in range(len(rating_count)):
    ratingCountKey.append(rating_count[i][0])

    rate = 0
    for j in [float(i) for i in rating_count[i][1]]:
        rate += j

    ratingCountValue.append(rate)


min_installKey = []
min_installValue = []
for i in range(len(min_install)):
    min_installKey.append(min_install[i][0])

    count = 0
    for j in [float(i) for i in min_install[i][1]]:
        if j != 0:
            count += 1

    min_installValue.append(count)


max_installKey = []
max_installValue = []
for i in range(len(max_install)):
    min_installKey.append(max_install[i][0])

    count = 0
    for j in [float(i) for i in max_install[i][1]]:
        if j != 0:
            count += 1

    max_installValue.append(count)


priceKey = []
priceValue = []
for i in range(len(price)):
    priceKey.append(price[i][0])

    amount = 0
    for j in [float(i) for i in price[i][1]]:
        amount += j

    priceValue.append(amount)


app = ratingKey
rating = ratingValue
countOfRating = ratingCountValue
mi_install = min_installValue
Price = priceValue

ma_install = max_installValue


x = []
y = []
for i in range(len(app)):
    x.append([float(rating[i]), float(countOfRating[i]), float(mi_install[i]), float(Price[i])])
    y.append(float(ma_install[i]))

learningRate = 0.000001
num_iteration = 100
size = len(y)
costs = []
beta = np.array([0.1, 0.1, 0.1, 0.1])

data = {'y':y, 'x':x}
df = DataFrame(data)
spark_df_from_pandas = spark.createDataFrame(df, schema=['x', 'y'])
myRDD=spark_df_from_pandas.rdd.map(lambda x: (float(x[0]), np.array(x[1])))

for i in range(num_iteration):
    gradientCost=myRDD.map(lambda x: (x[1], (x[0] - x[1] * beta)))\
                           .map(lambda x: (x[0]*x[1], x[1]**2 )).reduce(lambda x, y: (x[0] +y[0], x[1]+y[1] ))

    cost = 0
    for j in gradientCost[1]:
        cost += j

    gradient=(-1/float(size))* gradientCost[0]
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient

    costs.append(cost)

end2 = time.time()

print(f"Computation time of multi-linear regression by BGD is {(end2 - start2)/60} minutes")

xValues = [i for i in range(len(costs))]

plt.plot(xValues, costs, 'o', markersize=2)
plt.xlabel("Number of Iteration")
plt.ylabel("Cost")
plt.title("Cost with the number of iteration")
plt.show()