import random
import operator
from enum import Enum
import copy

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

NUM_DIMENSIONS = 2
NUM_CLUSTERS = 2
NUM_RANDOM_INSTANCES = 10
M = 2
E = 0.01

c = []
x = []
u = []
old_u = []

# Generate random datasets
for i in range(NUM_RANDOM_INSTANCES):
	inst = []
	for j in range(NUM_DIMENSIONS):
		inst.append(random.uniform(-10, 10))
	x.append(inst)

# Generate random clusters
for i in range(NUM_CLUSTERS):
	instance_index_as_cluster = random.randint(0, NUM_RANDOM_INSTANCES)
	c.append(x[instance_index_as_cluster])

# Initialize u
for i in range(len(x)):
	ui = []
	for j in range(NUM_CLUSTERS):
		ui.append(0)
	u.append(ui)
	old_u.append(ui)

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
for xi in x:
	print(xi)

print("\n================= CENTROID =================")
for cj in c:
	print(cj)

print("\n===================== U ====================")
for ui in u:
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
for old_ui in old_u:
	print(old_ui)
print("\n===================== U ====================")
for ui in u:
	print(ui)
