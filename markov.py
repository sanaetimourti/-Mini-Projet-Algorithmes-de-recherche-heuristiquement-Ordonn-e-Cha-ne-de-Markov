import numpy as np
import random
import networkx as nx
from collections import deque

def get_policy(env, goal):
    came_from = {}
    queue = deque([goal])
    visited = set([goal])
    came_from[goal] = None
    while queue:
        current = queue.popleft()
        for neigh in env.neighbors(current):
            if neigh not in visited:
                visited.add(neigh)
                queue.append(neigh)
                came_from[neigh] = current
    policy = {}
    for state in came_from:
        if state != goal:
            next_state = came_from[state]
            dx = next_state[0] - state[0]
            dy = next_state[1] - state[1]
            policy[state] = (dx, dy)
    return policy

def build_transition_matrix(env, policy, epsilon):
    states = [(x,y) for x in range(env.width) for y in range(env.height) if env.passable((x,y))]
    states.append('FAIL')
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((n_states, n_states))
    goal_idx = state_to_idx[env.goal]
    fail_idx = state_to_idx['FAIL']
    P[goal_idx, goal_idx] = 1.0
    P[fail_idx, fail_idx] = 1.0
    
    for state in states[:-1]:
        if state == env.goal:
            continue
        i = state_to_idx[state]
        action = policy.get(state, (0,0))
        slips = [(0,1),(0,-1)] if action[0] != 0 else [(1,0),(-1,0)]
        transitions = [(action, 1-epsilon), (slips[0], epsilon/2), (slips[1], epsilon/2)]
        
        for move, prob in transitions:
            nx, ny = state[0] + move[0], state[1] + move[1]
            next_state = (nx, ny) if env.in_bounds((nx,ny)) and env.passable((nx,ny)) else 'FAIL'
            j = state_to_idx[next_state]
            P[i, j] += prob
            
    assert np.allclose(P.sum(axis=1), 1.0)
    return P, states, state_to_idx

def compute_pi_n(P, states, start_state, goal_state, n_steps):
    idx = {s: i for i, s in enumerate(states)}
    pi_0 = np.zeros(len(states))
    pi_0[idx[start_state]] = 1.0
    P_n = np.linalg.matrix_power(P, n_steps)
    pi_n = pi_0 @ P_n
    return pi_n[idx[goal_state]]

def analyze_absorption(P, states, start_state, goal_state):
    idx = {s: i for i, s in enumerate(states)}
    absorbing = [goal_state, 'FAIL']
    transient = [s for s in states if s not in absorbing]
    
    if not transient:
        return 1 if start_state == goal_state else 0, 0
        
    num_t = len(transient)
    num_a = 2
    t_idx = {transient[k]: k for k in range(num_t)}
    a_idx = {absorbing[k]: k for k in range(num_a)}
    Q = np.zeros((num_t, num_t))
    R = np.zeros((num_t, num_a))
    
    for i in range(num_t):
        s = transient[i]
        for j in range(len(states)):
            t = states[j]
            p = P[idx[s], j]
            if t in transient:
                Q[i, t_idx[t]] = p
            else:
                R[i, a_idx[t]] = p
                
    I = np.eye(num_t)
    N = np.linalg.inv(I - Q)
    B = N @ R
    expected_steps = N.sum(axis=1)
    start_t = t_idx[start_state]
    prob_goal = B[start_t, 0]
    mean_time = expected_steps[start_t]
    return prob_goal, mean_time

def simulate_monte_carlo(env, P, states, start_state, goal_state, N_simulations=2000, max_steps=50):
    idx = {s: i for i, s in enumerate(states)}
    goal_idx = idx[goal_state]
    fail_idx = idx['FAIL']
    start_idx = idx[start_state]
    success_count = 0
    success_times = []
    all_times = []
    
    for _ in range(N_simulations):
        current = start_idx
        steps = 0
        while steps < max_steps and current not in [goal_idx, fail_idx]:
            current = random.choices(range(len(states)), weights=P[current])[0]
            steps += 1
        all_times.append(steps)
        if current == goal_idx:
            success_count += 1
            success_times.append(steps)
            
    prob_success = success_count / N_simulations
    mean_success_time = np.mean(success_times) if success_times else np.inf
    mean_total_time = np.mean(all_times)
    return prob_success, mean_success_time, mean_total_time

# Ajouts pour le graphe, classes et périodicité
def build_transition_graph(P, states):
    G = nx.DiGraph()
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if P[i, j] > 0:
                G.add_edge(from_state, to_state, weight=P[i, j])
    return G

def identify_classes(G):
    # Composantes fortement connexes pour les classes de communication
    classes = list(nx.strongly_connected_components(G))
    return classes

def check_periodicity(G):
    # Vérification de la périodicité pour chaque composante
    periodicities = {}
    for component in nx.strongly_connected_components(G):
        subgraph = G.subgraph(component)
        if len(component) > 1:
            cycles = list(nx.simple_cycles(subgraph))
            if cycles:
                lengths = [len(c) for c in cycles]
                gcd = np.gcd.reduce(lengths)
                periodicities[tuple(component)] = gcd
            else:
                periodicities[tuple(component)] = 1  # Apériodique si pas de cycles
        else:
            periodicities[tuple(component)] = 1
    return periodicities