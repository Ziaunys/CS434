from itertools import imap, izip, repeat
from math import sqrt
from operator import mul, sub
import random
import time

f = open("cluster-data.csv")
data = [tuple(map(float,line.strip('\r\n').split(','))) for line in f]

def find_centroid(cluster):
    avg = lambda x: sum(x)/len(x)
    return tuple(imap(avg, izip(*cluster)))

def find_clusters(data, centers):
    diff_sqr = lambda x, y: abs(x-y)**2
    dist = lambda x1, x2: sum(imap(diff_sqr, x1, x2))
    assign = lambda centers, point: min([(dist(center, point), center, point) for center in centers])
    clusters = [assign(centers, point) for point in data]
    return [[point[-1] for point in clusters if point[1] == center] for center in centers]

def kmeans(data, k = 3):
    diff_sqr = lambda x, y: abs(x-y)**2
    dist = lambda x1, x2: sum(imap(diff_sqr, x1, x2))**2
    centers = []
    past_centers = []
    while len(centers) < k:
        point = random.choice(data)
        if point not in centers:
            centers.append(point)
    new_centers = []
    SSE = []
    while set(new_centers) != set(centers):
        _SSE = 0
        if new_centers:
            past_centers.append(centers)
            centers = new_centers
        clusters = find_clusters(data, centers)
        new_centers = [find_centroid(cluster) for cluster in clusters]
        SSE.append(sum([sum(imap(dist, cluster, repeat(center))) for center, cluster in izip(new_centers, clusters)]))
    return clusters, SSE

cls, SSE = kmeans(data)
import matplotlib.pyplot as plt

plt.title("SSE for %s iterations of kmeans with k=3"%len(SSE))
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.plot(range(len(SSE)),SSE)
plt.show()
"""
plt.scatter([ i[0] for i in S[0]],[ i[1] for i in S[0]], c='b', marker='o')
plt.scatter([ i[0] for i in S[1]],[ i[1] for i in S[1]], c='r', marker='o')
plt.scatter([ i[0] for i in S[2]],[ i[1] for i in S[2]], c='g', marker='o')
plt.title('Kmeans clustering from Algorithm')
plt.show()
"""
SSE = []
for i in range(2,7):
    min_SSE = []
    for z in range(10):
        _, past_centers, final_centers = kmeans(data,k = i)
        for past_center in past_centers:
            min_SSE.append(sum([(c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 for c1, c2 in izip(past_center,final_centers)]))
    SSE.append(min(min_SSE))
plt.title("Minimum SSE for means k = 2..6")
plt.xlabel("k")
plt.ylabel("SSE")
plt.plot(range(2,7),SSE)
plt.show()
