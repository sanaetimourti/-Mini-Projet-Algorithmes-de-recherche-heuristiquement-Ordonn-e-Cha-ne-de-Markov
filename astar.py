import heapq
import itertools
import time

def manhattan(p, goal):
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

def zero_heuristic(p, goal):
    return 0

def graph_search(env, heuristic_func=manhattan, weight_g=1, weight_h=1):
    start = env.start
    goal = env.goal
    counter = itertools.count()
    open_list = []
    h_start = heuristic_func(start, goal)
    f_start = weight_g * 0 + weight_h * h_start
    heapq.heappush(open_list, (f_start, next(counter), start, None, 0))
    came_from = {}
    g_scores = {start: 0}
    nodes_expanded = 0
    max_open_size = 0
    start_time = time.time()
    closed_set = set()

    while open_list:
        f, _, current, parent, g = heapq.heappop(open_list)
        if current in closed_set:
            continue
        closed_set.add(current)
        came_from[current] = parent
        nodes_expanded += 1
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
            elapsed_time = time.time() - start_time
            return path, g, nodes_expanded, elapsed_time, max_open_size
            
        for neighbor in env.neighbors(current):
            tentative_g = g + env.cost
            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                h = heuristic_func(neighbor, goal)
                f_score = weight_g * tentative_g + weight_h * h
                heapq.heappush(open_list, (f_score, next(counter), neighbor, current, tentative_g))
                max_open_size = max(max_open_size, len(open_list))
                
    elapsed_time = time.time() - start_time
    return None, float('inf'), nodes_expanded, elapsed_time, max_open_size