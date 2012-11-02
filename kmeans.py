from itertools import imap, izip
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
    centers = []
    past_centers = []
    while len(centers) < k:
        point = random.choice(data)
        if point not in centers:
            centers.append(point)
    new_centers = []
    while set(new_centers) != set(centers):
        if new_centers:
            past_centers.append(centers)
            centers = new_centers
        clusters = find_clusters(data, centers)
        new_centers = [find_centroid(cluster) for cluster in clusters]
    return clusters, past_centers, centers

_, past_centers, final_centers = kmeans(data)
import matplotlib.pyplot as plt
SSE = []
for past_center in past_centers:
    SSE.append(sum([(c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 for c1, c2 in izip(past_center,final_centers)]))
plt.title("SSE for %s iterations of kmeans with k=3"%len(SSE))
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.plot(range(len(SSE)),SSE)
plt.show()


