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
        for i in range(1, n - 2):  # skip Alger
            for j in range(i+1, n-1):
                if j - i == 1: continue  
                before = dist_matrix[route[i-1], route[i]] + dist_matrix[route[j], route[j+1]]
                after = dist_matrix[route[i-1], route[j]] + dist_matrix[route[i], route[j+1]]
                if after < before:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
    return route

local_route = two_opt(start_route.copy())
local_dist = route_distance(local_route)
print('Local Search best distance:', local_dist)
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


#Recherche Recuit-Simulé
def simulated_annealing(start_route, initial_temp=1000, cooling_rate=0.995, min_temp=1, max_iter=1000):
    current_route = start_route.copy()
    current_distance = route_distance(current_route)
    best_route = current_route.copy()
    best_distance = current_distance
    
    temp = initial_temp
    iteration = 0
    while temp > min_temp and iteration < max_iter:
        iteration += 1
        # Generate neighbor by swapping two cities except algiers
        i, j = random.sample(range(1, len(current_route) - 1), 2)
        neighbor_route = current_route.copy()
        neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
        
        neighbor_distance = route_distance(neighbor_route)
        delta = neighbor_distance - current_distance
        
        # Accept neighbor if better or with probability exp(-delta / temp)
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_route = neighbor_route
            current_distance = neighbor_distance
            
            # Record best found
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
        
        # Cool temperature
        temp *= cooling_rate
        
    return best_route, best_distance

sa_route, sa_dist = simulated_annealing(start_route)
print("Simulated Annealing best distance:", sa_dist)
print("Simulated Annealing best route:", [cities.iloc[i]['city'] for i in sa_route])


#Recherche Tabu search.
def tabu_search(start_route, tabu_size=10, max_iter=500):
    current_route = start_route.copy()
    current_distance = route_distance(current_route)
    best_route = current_route.copy()
    best_distance = current_distance
    tabu_list = []
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        neighbors = []
        moves = []
        
        # Generate neighbors by swapping two cities (except start/end)
        for i in range(1, len(current_route) - 2):
            for j in range(i + 1, len(current_route) - 1):
                neighbor = current_route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
                moves.append((i,j))
        
        # Filter neighbors by tabu list except if better than best_distance (aspiration)
        best_candidate = None
        best_candidate_dist = float('inf')
        best_move = None
        
        for idx, neighbor in enumerate(neighbors):
            dist = route_distance(neighbor)
            move = moves[idx]
            if (move not in tabu_list) or (dist < best_distance):
                if dist < best_candidate_dist:
                    best_candidate = neighbor
                    best_candidate_dist = dist
                    best_move = move
        
        if best_candidate is None:  # no admissible moves
            break
        
        current_route = best_candidate
        current_distance = best_candidate_dist
        
        # Update tabu list (FIFO)
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        
        # Update best solution found
        if current_distance < best_distance:
            best_route = current_route.copy()
            best_distance = current_distance
    
    return best_route, best_distance

ts_route, ts_dist = tabu_search(start_route)
print("Tabu Search best distance:", ts_dist)
print("Tabu Search best route:", [cities.iloc[i]['city'] for i in ts_route])


#Recherche par Algorithme génétique
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(1, size - 1), 2))
    
    child = [None] * size
    child[0], child[-1] = parent1[0], parent1[-1]
    
    # Copie segment parent1
    for i in range(start, end + 1):
        child[i] = parent1[i]
    
    # Liste villes parent2 SANS Algiers
    p2_cities = parent2[1:-1]
    
    # Remplissage circulaire garanti
    child_pos = 1
    p2_pos = 0
    while child_pos < size - 1:
        if child[child_pos] is None:
            while p2_cities[p2_pos % len(p2_cities)] in child:
                p2_pos += 1
            child[child_pos] = p2_cities[p2_pos % len(p2_cities)]
            p2_pos += 1
        child_pos += 1
    
    return child


def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route) - 1), 2)
        route[i], route[j] = route[j], route[i]
    return route


def genetic_algorithm(pop_size=50, generations=200):
    population = [random_route() for _ in range(pop_size)]
    
    for gen in range(generations):
        population = sorted(population, key=route_distance)
        new_population = population[:int(pop_size*0.2)]  # elitism 20%
        
        while len(new_population) < pop_size:
            parents = random.sample(population[:25], 2)  # top 50%
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    best_individual = min(population, key=route_distance)
    best_distance = route_distance(best_individual)
    return best_individual, best_distance


ga_route, ga_dist = genetic_algorithm()
print("Genetic Algorithm best distance:", ga_dist)
print("Genetic Algorithm best route:", [cities.iloc[i]['city'] for i in ga_route])
