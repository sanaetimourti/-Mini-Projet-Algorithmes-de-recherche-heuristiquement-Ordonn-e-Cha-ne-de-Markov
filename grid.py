import networkx as nx
import matplotlib.pyplot as plt

class GridEnv:
    def __init__(self, width, height, start, goal, obstacles, epsilon=0.0):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.epsilon = epsilon
        self.cost = 1

    def in_bounds(self, state):
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, state):
        return state not in self.obstacles

    def neighbors(self, state):
        x, y = state
        results = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [pos for pos in results if self.in_bounds(pos) and self.passable(pos)]

def create_experiment_grids():
    grids = {}
    grids['facile'] = GridEnv(5, 5, (0, 0), (4, 4), [])
    grids['moyenne'] = GridEnv(5, 5, (0, 0), (4, 4), [(2,1), (2,2), (2,3)])
    grids['difficile'] = GridEnv(5, 5, (0, 0), (4, 4), [(1,1), (1,2), (1,3), (2,3), (3,1), (3,2)])
    return grids