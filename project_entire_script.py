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


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: wordcount <file> <output> ", file=sys.stderr)
		exit(-1)

	spark = SparkSession.builder.master("local[*]").getOrCreate()
	sc = SparkContext.getOrCreate()
	sqlContext = SQLContext(sc)
	
	# load data set
	lines2 = sc.textFile("Google-Playstore.csv")
	
	# generate test case file
	df = pd.read_csv("Google-Playstore.csv")
	test_case = df.sample(n = 10000)
	test_case.to_csv('Google-Playstore_test.csv', index=False)
	
	"""
	
	Simple Linear Regression
	
	"""
	
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
	
	
	"""
	
	Gradient Descent for parameters
	
	"""
	
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
	
	
	"""
	
	Multi-Linear Regression
	
	"""
	
	print("##### Finding the parameters of multi-linear regression using gradient descent #####")
	
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
	
	
	"""
	
	Logistic Regression
	
	"""
		
	print("##### gradient descent algorithm to learn a logistic regression model #####")
	
	start3 = time.time()
	
	total_install = cleaned.map(lambda p: (p[0], p[7]))
	tuples = total_install.collect()

	appWords = []

	for i in app:
		words = i.split(" ")
		for j in words:
			j = re.sub('[^A-Za-z0-9]+', '', j)
			appWords.append(j)

	appWords = ' '.join(appWords).split()

	allWords = sc.parallelize(appWords)
	allCount = allWords.map(lambda x: (x, 1)).reduceByKey(add)
	topWords = allCount.top(20000, lambda x: x[1])

	topWordsK = sc.parallelize(range(20000))
	dictionary = topWordsK.map(lambda x: (topWords[x][0], x))

	def TF(words_list, top_words):
		words_dict = dict(Counter(words_list))
		tf_vector = []
		for word in top_words:
			if word in words_dict.keys():
				tf = words_dict[word]
				tf_vector.append(tf)
			else:
				tf_vector.append(0)
		return tf_vector

	key_id = [i for i in range(len(tuples))]
	key_values = {tuples[i][0]: i for i in key_id}
	topWordsBC = sc.broadcast(dictionary.keys().collect())
	feat = cleaned.map(lambda x: (key_values[x[0]], TF(x[1], topWordsBC.value)))
	labels = cleaned.map(lambda x: (key_values[x[0]], int(x[0][0] == 'A' and x[0][1] == 'U')))
	trainRDD = feat.join(labels)

	learningRate = 0.0003
	num_iteration = 5
	lambda_cof = 0.01
	size = len(tuples)

	loss_list = list()

	parameter_vector = np.random.normal(0, 0.1, (dictionary.count(), 1))
	
	def sigmoid(x):
		return 1.0 / (1 + np.exp(-x))

	def loss_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		return -y * np.log(pred + 1e-12) - (1 - y) * np.log((1 - pred) + 1e-12)

	def accuracy_score(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		pred = 1 if pred >= 0.5 else 0
		acc = int(pred == y)
		return acc

	def grad_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		grad = (pred - y) @ feat_line[None, :]
		return grad

	acc_his = []
	loss_his = []
	grad_his = []
	prev_param_norm = 0
	
	for i in tqdm.trange(num_iteration):
		parameter_vector_BC = sc.broadcast(parameter_vector)

		loss = trainRDD.map(lambda x: loss_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add) / size
		acc = trainRDD.map(lambda x: accuracy_score(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add) / size
		grad = trainRDD.map(lambda x: grad_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add) / size

		parameter_vector = parameter_vector - learningRate * grad[:, None]

		if np.abs(np.linalg.norm(parameter_vector) - prev_param_norm) < 1e-7:
			print('Break')
			break

		prev_param_norm = np.linalg.norm(parameter_vector)
		# L2
		parameter_vector = parameter_vector - 2 * lambda_cof * parameter_vector

		acc_his.append(acc)
		loss_his.append(loss)
		grad_his.append(np.linalg.norm(grad))
	
	end3 = time.time()
	
	print(f"Computation time of multi-linear regression by BGD is {(end3 - start3)/60} minutes")

	fig, ax = plt.subplots(3, figsize=(13, 13))
	ax[0].set_title('Accurary')
	ax[0].plot(acc_his)
	ax[1].set_title('Loss')
	ax[1].plot(loss_his)
	ax[2].set_title('GradNorm')
	ax[2].plot(grad_his)
	plt.savefig("TrainingProcess.png")
	plt.show()

	print('The five words with the largest coefficients',
		  np.array(topWordsBC.value)[np.argsort(parameter_vector[:, 0])[-5:]])
	
	
	"""
	
	Logistic Regression Model Evaluation
	
	"""
		
	print("##### model evaluation #####")
	
	start4 = time.time()
	
	t_tuples = total_install.collect()
	key_id = [i for i in range(len(tuples))]
	key_values = {tuples[i][0]: i for i in key_id}
	topWordsBC = sc.broadcast(dictionary.keys().collect())
	feat = cleaned.map(lambda x: (key_values[x[0]], TF(x[1], topWordsBC.value)))
	labels = cleaned.map(lambda x: (key_values[x[0]], int(x[0][0] == 'A' and x[0][1] == 'U')))
	testRDD = feat.join(labels)


	# val
	def TP_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		pred = 1 if pred >= 0.5 else 0
		TP = int(pred == 1 and y == 1)
		return TP


	def FP_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		pred = 1 if pred >= 0.5 else 0
		FP = int(pred == 1 and y != 1)
		return FP


	def FN_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		pred = 1 if pred >= 0.5 else 0
		FN = int(pred != 1 and y == 1)
		return FN


	def TN_func(feat_line, y, parameter_vector):
		feat_line = np.array(feat_line)
		pred = sigmoid(np.dot(feat_line, parameter_vector))
		pred = 1 if pred >= 0.5 else 0
		TN = int(pred != 1 and y != 1)
		return TN


	parameter_vector_BC = sc.broadcast(parameter_vector)
	acc = testRDD.map(lambda x: accuracy_score(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add) / size
	TP = testRDD.map(lambda x: TP_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add)
	FP = testRDD.map(lambda x: FP_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add)
	FN = testRDD.map(lambda x: FN_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add)
	TN = testRDD.map(lambda x: TN_func(x[1][0], x[1][1], parameter_vector_BC.value)).reduce(add)

	F1 = 2 * TP / (2 * TP + FN + FP)
	print('The Acc of Test: ', acc)
	print('The F1 score of Test: ', F1)
	
	print(f"Computation time of multi-linear regression by BGD is {(end4 - start4)/60} minutes")
		
		
	"""
	
	SVM Model
	
	"""
		
	print("##### SVM model #####")
	
	start5 = time.time()
	
	d_corpus = sc.textFile("Google-Playstore.csv")
	d_keyAndText = d_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	d_keyAndListOfWords = d_keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split())).sortByKey(False)

	tuples = d_keyAndListOfWords.collect()
	allWordsList = []

	for i in range(len(tuples)):
		for j in tuples[i][1]:
			allWordsList.append(j)

	allWords = sc.parallelize(allWordsList)
	allCount = allWords.map(lambda x: (x, 1)).reduceByKey(add)
	topWords = allCount.top(20000, lambda x: x[1])

	topWordsK = sc.parallelize(range(20000))
	dictionary = topWordsK.map(lambda x: (topWords[x][0], x))


	def TF(words_list, top_words):
		words_dict = dict(Counter(words_list))
		tf_vector = []
		for word in top_words:
			if word in words_dict.keys():
				tf = words_dict[word]
				tf_vector.append(tf)
			else:
				tf_vector.append(0)
		return Vectors.dense(tf_vector)

	key_id = [i for i in range(len(tuples))]
	key_values = {tuples[i][0]: i for i in key_id}
	topWordsBC = sc.broadcast(dictionary.keys().collect())
	feat = d_keyAndListOfWords.map(lambda x: (key_values[x[0]], TF(x[1], topWordsBC.value)))
	labels = d_keyAndListOfWords.map(lambda x: (key_values[x[0]], int(x[0][0] == 'A' and x[0][1] == 'U')))
	train_feat_df = sqlContext.createDataFrame(feat, ['ind', 'features'])
	train_labels_df = sqlContext.createDataFrame(labels, ['ind', 'labels'])
	train_df = train_feat_df.join(train_labels_df, on=['ind']).sort(['ind'])
	train_df.cache()

	svc = LinearSVC(labelCol='labels')
	svg_model = svc.fit(train_df)

	st_test_read = time.time()
	t_corpus = sc.textFile('Google-Playstore_test.csv')
	t_keyAndText = t_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')
	t_keyAndListOfWords = t_keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split())).sortByKey(False)
	t_tuples = t_keyAndListOfWords.collect()
	t_key_id = [i for i in range(len(t_tuples))]
	t_key_values = {t_tuples[i][0]: i for i in t_key_id}
	t_feat = t_keyAndListOfWords.map(lambda x: (t_key_values[x[0]], TF(x[1], topWordsBC.value)))
	t_labels = t_keyAndListOfWords.map(
		lambda x: (t_key_values[x[0]], int(x[0][0] == 'A' and x[0][1] == 'U')))
	test_feat_df = sqlContext.createDataFrame(t_feat, ['ind', 'features'])
	test_labels_df = sqlContext.createDataFrame(t_labels, ['ind', 'labels'])
	test_df = test_feat_df.join(test_labels_df, on=['ind']).sort(['ind'])
	test_df.cache()

	st_test = time.time()
	test_pred = svg_model.evaluate(test_df).predictions
	evaluator = MulticlassClassificationEvaluator(labelCol='labels')

	print('Acc of Test: ', evaluator.evaluate(test_pred, {evaluator.metricName: "accuracy"}))
	print('F1 of Test:', evaluator.evaluate(test_pred, {evaluator.metricName: "f1"}))
	
	end5 = time.time()

	print(f"Computation time of SVM regression by BGD is {(end5 - start5)/60} minutes")
	