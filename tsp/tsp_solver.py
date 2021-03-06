import random
from functools import reduce 
import argparse
import csv

import tsplib95 

log = print
log = lambda x: None
class Graph():
    class Edge():
        def __init__(self, graph, src, dst, weight):
            self.graph = graph
            self.src = src
            self.dst = dst
            self.weight = weight
            self.pheromone = 1.0
            self.importance = None
            self.probability = None

        def set_pheromone(self, pheromone):
            self.pheromone = pheromone
            self.graph.edges[self.dst][self.src].pheromone = pheromone
        
        def set_importance(self, importance):
            self.importance = importance
            self.graph.edges[self.dst][self.src].importance = importance

        def set_probability(self, probability):
            self.probability = probability
            self.graph.edges[self.dst][self.src].probability = probability

    def __init__(self, weight_matrix):
        self.edges = []
        for i in range(len(weight_matrix)):
            row_edges = []
            for j in range(len(weight_matrix[i])):
                edge = self.Edge(self, i, j, weight_matrix[i][j])
                row_edges.append(edge)
            self.edges.append(row_edges)
        self.nodes = range(len(weight_matrix))
        self.num_nodes = len(self.nodes)
    

class ACO():
    solution_filename = 'solution.csv'
    class Ant():
        def __init__(self, aco, graph):
            self.aco = aco
            self.graph = graph
            self.current_node = None
            self.visitables = []
            self.path = []
            self.edges = []
        
        def initialize(self, start_node):
            self.current_node = start_node
            self.visitables = list(self.graph.nodes)
            self.visitables.remove(start_node)
            self.path = [start_node]
            self.edges = []

        def select_next_edge(self, start_node):
            edges = [self.graph.edges[start_node][dst] for dst in self.visitables]
            
            if random.random() < self.aco.selection_probability:
                next_edge = max(edges, key=lambda x: x.importance)
            else:
                sum_importances = sum(map(lambda y: y.importance, edges))
                [edge.set_probability(edge.importance / sum_importances if sum_importances else 1) for edge in edges]
                probabilities = [edge.probability for edge in edges]
                next_edge = random.choices(edges, weights=probabilities, k=1)[0]
            return next_edge 

        def visit_node(self, node):
            self.visitables.remove(node)

        def evaluate(self):
            weights = map(lambda x: x.weight, self.edges)
            cost = reduce(lambda x, y: x + y, weights)
            return cost

        def explore(self, start_node):
            self.initialize(start_node)
            while self.visitables:
                next_edge = self.select_next_edge(self.current_node)                    
                next_node = next_edge.dst
                self.path.append(next_node)
                self.edges.append(next_edge)
                self.visit_node(next_node)
                self.current_node = next_node

            last_node = self.path[-1]
            self.edges.append(self.graph.edges[last_node][start_node])
            self.path.append(start_node)
            

    def __init__(self, num_ants, evaporation_rate, alpha, beta, selection_probability):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.selection_probability = selection_probability

    def _get_importance(self, pheromone, distance):
        heuristic = 1 / distance if distance != 0 else float('inf')
        return (pheromone ** self.alpha) * (heuristic ** self.beta)
    
    def _update_importance(self, edges):
        for row_edges in edges:
            for edge in row_edges:
                importance = self._get_importance(edge.pheromone, edge.weight)
                edge.set_importance(importance)

    def _evaporate(self, edges):
        [edge.set_pheromone(edge.pheromone * (1 - self.evaporation_rate)) for edge in edges]

    def _add_pheromone(self, edges, cost):
        [edge.set_pheromone(edge.pheromone + 1 / cost) for edge in edges]
    
    def _save_solution(self, best_path, csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for node in best_path[:-1]: # ?
                writer.writerow([str(node+1)])
        
    def optimize(self, graph, num_iter, stop_iter=None):
        stop_iter = num_iter / 2 if stop_iter is None else stop_iter
        best_fitness = float('inf')
        best_path = None
        best_edges = None
        self._update_importance(graph.edges)
        unchanged_cnt = 0
        ants = [self.Ant(self, graph) for _ in range(self.num_ants)]
        for i in range(num_iter):
            log(f"Iteration {i}")
            best_iter_fitness = float('inf')
            for ant_no, ant in enumerate(ants):
                start_node = random.choice(graph.nodes)
                ant.explore(start_node)
                fitness = ant.evaluate()
                assert ant.path[0] == ant.path[-1]
                assert len(set(ant.path)) == len(graph.nodes)
                assert len(ant.path) == len(graph.nodes) + 1
    
                if fitness < best_fitness:
                    best_path = ant.path
                    best_edges = ant.edges
                    best_fitness = fitness
                    self._save_solution(best_path, self.solution_filename)
                if fitness < best_iter_fitness:
                    best_iter_fitness = fitness
                    
                self._add_pheromone(ant.edges, fitness)
            if best_iter_fitness == best_fitness:
                unchanged_cnt += 1
                if unchanged_cnt > stop_iter:
                    break
            else:
                unchanged_cnt = 0
            self._evaporate(sum(graph.edges, []))
            self._add_pheromone(best_edges, best_fitness)
            self._update_importance(graph.edges)
            log(f'Best cost: {best_fitness: .0f} path: {best_path}')
        return best_fitness, best_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tsp_filename', type=str, help="TSP file name ex) dj38.tsp")
    parser.add_argument('-p', type=int, default=50, help="Population size")
    parser.add_argument('-f', type=int, default=500, help="The total number of fitness evaluations")
    args = parser.parse_args()
    num_ants = args.p
    num_iter = args.f
    tsp_filename = args.tsp_filename

    aco = ACO(num_ants, 0.1, 10, 10, 0.1)
    prob = tsplib95.load(tsp_filename)
    log('Problem loaded')
    matrix = [
        [ prob.get_weight(src, dst) for dst in range(1, prob.dimension + 1)]
        for src in range(1, prob.dimension + 1) ]
    log('Matrix is converted')
    del prob
    graph = Graph(matrix)
    del matrix
    best_fitness, best_path = aco.optimize(graph, num_iter, None)
    print(f"{best_fitness:.0f}")

if __name__ == "__main__":
    main()