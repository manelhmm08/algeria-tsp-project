import pandas as pd
import numpy as np
import random


cities = pd.read_csv('algeria_20_cities_xy (1).csv')
n = len(cities)

#nhsbo Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#distance matrix
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = euclidean_distance(
            cities.iloc[i]["x_km"], cities.iloc[i]["y_km"],
            cities.iloc[j]["x_km"], cities.iloc[j]["y_km"]
        )


algiers_index = cities.index[cities['city'] == 'Algiers'][0]
other_cities = list(cities.index)
other_cities.remove(algiers_index)

def random_route():
    route = other_cities.copy()
    random.shuffle(route)
    route = [algiers_index] + route + [algiers_index]
    return route

def route_distance(route):
    return sum(dist_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

#Random Search
best_rand_route, best_rand_dist = None, float('inf')
for _ in range(1000):
    route = random_route()
    dist = route_distance(route)
    if dist < best_rand_dist:
        best_rand_route, best_rand_dist = route, dist
print('Random Search best distance:', best_rand_dist)
print('Random Search route:', [cities.iloc[i]['city'] for i in best_rand_route])

#Initial Route for local search and hill climbing
start_route = random_route()

#Local Search 
def two_opt(route):
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):  # skip Algiers at start/end
            for j in range(i+1, n-1):
                if j - i == 1: continue  # adjacent, skip
                before = dist_matrix[route[i-1], route[i]] + dist_matrix[route[j], route[j+1]]
                after = dist_matrix[route[i-1], route[j]] + dist_matrix[route[i], route[j+1]]
                if after < before:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
    return route

local_route = two_opt(start_route.copy())
local_dist = route_distance(local_route)
print('Local Search (2-opt) best distance:', local_dist)
print('Local Search Tour:', [cities.iloc[i]['city'] for i in local_route])

#Hill Climbing 
def get_neighbors(route):
    neighbors = []
    for i in range(1, len(route)-2):
        for j in range(i+1, len(route)-1):
            neighbor = route.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def hill_climbing(start_route):
    current_route = start_route.copy()
    current_distance = route_distance(current_route)
    while True:
        neighbors = get_neighbors(current_route)
        best_neighbor = min(neighbors, key=route_distance)
        best_neighbor_dist = route_distance(best_neighbor)
        if best_neighbor_dist < current_distance:
            current_route, current_distance = best_neighbor, best_neighbor_dist
        else:
            break
    return current_route, current_distance

hc_route, hc_dist = hill_climbing(start_route)
print('Hill climbing best distance:', hc_dist)
print('Hill climbing route:', [cities.iloc[i]['city'] for i in hc_route])
