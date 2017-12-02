import random
import operator
from enum import Enum
import copy

# read dataset
import importlib
import readtest

# uncomment this if you want to reload read.py
#read = importlib.reload(read)
# data_list = read.data_raw_numeric
data_list = readtest.data_raw_numeric
# read data train
num_train = 0.8 * len(data_list)
data_train = data_list[:int(num_train)]

# read data test
num_test = 0.2 * len(data_list)
data_test = data_list[int(num_train):]

class vector(Enum):
	# def dot(v1, v2):
	# 	result = 0
	# 	for vi in map(operator.mul, zip(v1,v2)):
	# 		result += vi
	# 	return result

	def add(v1, v2):
		result = []
		for i in range(len(v1)):
			result.append(v1[i] + v2[i])
		return result

	def mul(n, v):
		return [(vi * n) for vi in v]

	def div(n, v):
		return [(vi / n) for vi in v]


random.seed(0)

NUM_CLUSTERS = 2
NUM_RANDOM_INSTANCES = len(data_train)
LABEL = ['<=50K', '>50K']
M = 2
E = 0.1 ** 10

c = [data_train[0], data_train[1000]]
x = data_train
u = []
old_u = []

# Initialize u
for i in range(len(x)):
	ui = []
	for j in range(NUM_CLUSTERS):
		ui.append(0)
	u.append(ui)
	old_u.append(ui)

def predict(u):
	results = []
	for ui in u:
		maximum = -1
		j_max = -1
		for j, uij in enumerate(ui):
			if maximum < uij:
				maximum = uij
				j_max = j
		results.append((j_max, maximum))
	return results

def get_accuracy(tp, tn, fp, fn):
	try:
		return (tp+tn)/(tp+tn+fp+fn)
	except:
		return 0

def get_precision(tp, tn, fp, fn):
	try:
		return (tp)/(tp+fp)
	except:
		return 0

def get_recall(tp, tn, fp, fn):
	try:
		return (tp)/(tp+fn)
	except:
		return 0

def get_f1(precision, recall):
	try:
		return 2 * (precision*recall) / (precision+recall)
	except:
		return 0

def evaluate(predictions):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	idx_test = len(data_list) - len(data_test)
	for i, pred in enumerate(predictions):
		if LABEL[pred[0]] == ">50K":
			if (LABEL[pred[0]] == readtest.data_test.loc[i+idx_test]['income']):
				tp+=1
			else:
				fp+=1
		else: # LABEL[pred[0]] == "<=50K"
			if (LABEL[pred[0]] == readtest.data_test.loc[i+idx_test]['income']):
				tn+=1
			else:
				fn+=1
	print("tp:", tp)
	print("tn:", tn)
	print("fp:", fp)
	print("fn:", fn)
	accuracy = get_accuracy(tp, tn, fp, fn)
	precision = get_precision(tp, tn, fp, fn)
	recall = get_recall(tp, tn, fp, fn)
	f1 = get_f1(precision, recall)
	print("Accuracy	:", accuracy*100, "%")
	print("Precision	:", precision*100, "%")
	print("Recall		:", recall*100, "%")
	print("F1		:", f1*100, "%")
	return (tp, tn, fp, fn)

def get_similarity(xi, cj):
	sum_of_diff = 0
	for i in range(len(xi)):
		sum_of_diff += (xi[i] - cj[i]) ** 2
	return sum_of_diff ** (1/2)

def get_new_uij(i, j):
	result = 0
	for k, ck in enumerate(c):
		try:
			result += (
				get_similarity(x[i], c[j]) 
				/ 
				get_similarity(x[i], ck)
			)
		except:
			pass
	result **= (2/(M-1))
	try:
		return 1 / result
	except:
		return 0

def initialize_u():
	u = list()
	for i in range(len(x)):
		ui = []
		for j in range(NUM_CLUSTERS):
			ui.append(float(0))
		u.append(ui)

def update_u(**kwargs):
	global old_u, u
	old_u = copy.deepcopy(u)
	initialize_u()
	for j, cj in enumerate(c):
		for i, xi in enumerate(x):
			u[i][j] = get_new_uij(i, j)

def get_new_cj(j):
	result1 = [0] * len(x[0])
	result2 = 0
	for i, xi in enumerate(x):
		result1 = vector.add(
			result1,
			vector.mul(u[i][j] ** M, xi)
		)
	for i, xi in enumerate(x):
		result2 += u[i][j] ** M
	return vector.div(result2, result1)

def update_c(**kwargs):
	for j, cj in enumerate(c):
		c[j] = get_new_cj(j)

def is_converge():
	global u, old_u
	maximum = 0;
	for i, ui in enumerate(u):
		for j, uij in enumerate(ui):
			error = abs(uij - old_u[i][j])
			if error >= E:
				return False
	return True

def get_max_u():
	maximum = 0;
	for i, ui in enumerate(u):
		for j, uij in enumerate(ui):
			error = abs(uij - old_u[i][j])
			if error > maximum:
				maximum = error
	return maximum

update_u()

print("\n================== POINTS ==================")
for xi in x[:10]:
	print(xi)

print("\n================= CENTROID =================")
for cj in c:
	print(cj)

print("\n===================== U ====================")
for ui in u[:10]:
	print(ui)

iteration = 0
while(not is_converge()):
	iteration+=1
	print("#",iteration)
	update_c()
	update_u()
	print(get_max_u())

print("CONVERGE AT #", iteration)
print("\n=================== OLD U ==================")
for old_ui in old_u[:10]:
	print(old_ui)
print("\n===================== U ====================")
for ui in u[:10]:
	print(ui)

# TESTING
# initialization testing
xtest = data_test
utest = []
old_utest = []

# function for testing
for i in range(len(xtest)):
	ui = []
	for j in range(NUM_CLUSTERS):
		ui.append(0)
	utest.append(ui)
	old_utest.append(ui)

def get_new_uij_test(i, j):
	result = 0
	for k, ck in enumerate(c):
		try:
			result += (
				get_similarity(xtest[i], c[j]) 
				/ 
				get_similarity(xtest[i], ck)
			)
		except:
			pass
	result **= (2/(M-1))
	try:
		return 1 / result
	except:
		return 0

def initialize_u_test():
	utest = list()
	for i in range(len(xtest)):
		ui = []
		for j in range(NUM_CLUSTERS):
			ui.append(float(0))
		utest.append(ui)

def update_utest(**kwargs):
	global old_utest, utest
	old_utest = copy.deepcopy(utest)
	initialize_u_test()
	for j, cj in enumerate(c):
		for i, xi in enumerate(xtest):
			utest[i][j] = get_new_uij_test(i, j)

update_utest()
# print(len(utest))
# print("\n===================== U ====================")
# for ui in utest[:10]:
# 	print(ui)

preds = predict(utest)
for i in preds[:10]:
	print(i)
evaluate(preds)