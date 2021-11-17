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

reg = LinearRegression(fit_intercept=True).fit(X, y)

print("The m (coefficient) =",reg.coef_)
print("The b (y-intercept) =",reg.intercept_)
print("The equation is: y = "+str(reg.coef_[0])+"X + "+str(reg.intercept_))

print("##### Finding the parameters using gradient descent #####")

start1 = time.time()
df = np.stack([y, x], axis=1)
dff = map(lambda x: (float(x[0]), Vectors.dense(x[1:])), df)
mydf = spark.createDataFrame(dff, schema=["Money", "Distance"])
myRDD=mydf.rdd.map(tuple).map(lambda x: (float(x[0]), np.array(x[1]) ))

learningRate = 0.00001
num_iteration = 100
size = float(len(y))
beta = np.array([0.1])
costs = []

for i in range(num_iteration):
    gradientCost=myRDD.map(lambda x: (x[1], (x[0] - x[1] * beta) ))\
                           .map(lambda x: (x[0]*x[1], x[1]**2 )).reduce(lambda x, y: (x[0] +y[0], x[1]+y[1] ))
    cost= gradientCost[1]
    gradient=(-1/float(size))* gradientCost[0]
    print(i, "Beta", beta, " Cost", cost)
    beta = beta - learningRate * gradient
    costs.append(cost[0])

end1 = time.time()

print(f"Computation time of BGD is {(end1 - start1)/60} Minutes")

# making plot
xValues = [i for i in range(len(costs))]
plt.plot(xValues, costs, 'o', markersize=2)
plt.xlabel("Number of Iteration")
plt.ylabel("Cost")
plt.title("Cost with the number of iteration")
plt.show()
