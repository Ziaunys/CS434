from __future__ import division
from operator import mul
from itertools import imap, izip, repeat
from numpy import matrix
import numpy as np

f = open('tn-train.txt')
data = [tuple(map(float, line.split('\t'))) for line in f if line]
f = open('tn-test.txt')
data_test = [tuple(map(float, line.split('\t'))) for line in f if line]

y = [170, 191, 189, 180.34, 171, 176.53, 187, 185.42, 190, 181, 180, 175, 185.42,190,181,180,175, 188, 170, 185]
x1 = [50, 57, 50, 53.34, 54, 55.88, 57, 55.88, 57, 54, 55, 53, 55.88, 57, 54, 55, 53, 57, 49.5, 57]
x2 = [166, 196, 191, 180.34, 174, 176.53, 177, 208.28, 199, 181, 178, 172, 208.28,199, 181, 178, 172, 185, 165, 188]


xs = [list(repeat(1,len(x1))) ,x1, x2]

"""
Given a list of input vectors and the corresponding output learn a linear regression
and return a function which predicts y given 2 inputs
"""
def multi_regression(xs,y):
    xs = matrix(xs)
    xTy = np.dot(xs,y)
    xTx = np.dot(xs,xs.T)
    w = np.dot(xTy,xTx.I)
    w = np.array(w).reshape(-1,).tolist()
    print type(w)
    print type(w[0])
    return lambda x1,x2 : w[0] + x1*w[1] + x2*w[2]

"""
Given a list of x and y values return a function which predicts y for a given x
"""
def linear_regression(x, y):
    avg = lambda data: sum(data) / len(data)
    xy_bar = avg(map(mul, x, y))
    x_bar = avg(x)
    y_bar = avg(y)
    x_sq_bar = avg(map(pow, x, repeat(2,len(x))))
    w = (xy_bar - x_bar*y_bar) / (x_sq_bar - x_bar**2)
    b = y_bar - w * x_bar
    return lambda x: w*x + b


"""
Tests and SSE

In our document we misunderstood the SSE who our values are too low as we divided by the number of tests
so the SSE is off by a factor of the len(y)
"""


f = multi_regression(xs,y)
sse = 0
for xi,xii,output in zip(x1,x2,y):
    sse += (f(xi,xii) - output)**2
print "x1 x2 sse"
print sse/len(y)

f = linear_regression(x1,y)
sse = 0
for xi,yi in zip(x1,y):
    sse += (f(xi) - yi)**2
#    print "%.3f & %.3f & %.3f & %.3f\\\\"%(xi,yi,f(xi),abs(f(xi) - yi))
print "x1 sse1"
print sse/len(y)
f = linear_regression(x2,y)
print "x2 sse1"
sse = 0
for xi, yi in zip(x2,y):
    sse+= (f(xi) - yi)**2
#    print "%.3f & %.3f & %.3f & %.3f \\\\"%(xi,yi,f(xi),abs(f(xi) - yi))
print sse/len(y)
