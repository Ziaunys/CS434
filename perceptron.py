from itertools import imap, izip, repeat
from operator import mul, pow
import matplotlib.pyplot as plt
import random

f = open('tn-train.txt')
data = [tuple(map(float, line.split('\t'))) for line in f if line]
f = open('tn-test.txt')
data_test = [tuple(map(float, line.split('\t'))) for line in f if line]
y = map(lambda x: x[0], data)
xs = map(lambda x: (1, x[1], x[2]), data)
y_test = map(lambda x: x[0], data_test)
xs_test = map(lambda x: (1, x[1], x[2]), data_test)

def p_train(xs,ys,shuffled=False,epoch=0):
	w = list(repeat(0, len(xs[0])))
	dot_product = lambda x, w: sum(imap(mul, x, w))
	updated = True
	data = zip(xs,ys)
	if shuffled:
		random.shuffle(data)
	if epoch:
		data = data[:epoch]
	while updated:
		if shuffled:
			random.shuffle(data)
		updated = False
		for x,y in data:
			print y
			if dot_product(x, w) * y <= 0:
				w = [y * xi + wi for xi, wi in izip(x, w)]
				updated = True
	return lambda x: -(w[0] + x * w[1]) / w[2]	

x = range(-4,9)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)
plt.title('Learned Linear Classifier: Learning Testing Data Unshuffled')

line_f = p_train(xs_test,y_test,epoch=50)
y1 = map(line_f, x)
plt.plot(x,y1,label="Linear Classifier with 50 epochs")
plt.show()
line_f = p_train(xs_test,y_test, epoch=75)
y2 = map(line_f, x)
plt.plot(x,y2,label="Linear Classifier with 75 epochs")
plt.show()
line_f = p_train(xs_test,y_test, epoch=100)
y3 = map(line_f, x)
plt.plot(x,y3,label="Linear Classifier with 100 epochs")
plt.show()
plus = [input[1:] for input in data_test if input[0] > 0]
minus = [input[1:] for input in data_test if input[0] < 0]

x1 , y1 = zip(*plus)
plt.scatter(x1, y1, s=20, c='b', marker="s")
x1 , y1 = zip(*minus)
plt.scatter(x1, y1, s=20, c='r', marker="o")
plt.legend()
plt.show()



