
#from tokenize import Double
import numpy as np
#from itertools import product

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from fitnessBDAL import Fitness



#from collections import deque, defaultdict
#from ast2json import ast2json
import time

class TestGenerator:

    def __init__(self,SUTpath="",algorithm={},n_var=2,multipath=True,verbose=False):
        self.fitness = Fitness(SUTpath=SUTpath,n_var=n_var)
        if algorithm == {}:
            algorithm = PSO(pop_size=10)
        self.algorithm = algorithm
        self.generate()
    
    def generate(self):
        st = time.time()
        best_positions = {}
        more_paths = True
        past_walking = []
        while more_paths:
            res = minimize(self.fitness,
                self.algorithm,
                seed=1,
                verbose=False)
            cost = res.F[0];pos = res.X

            particle_pos = np.array([pos],np.float32)
            self.fitness.resolve_path(particle_pos)
            has_path = lambda x: lambda y: y in x
            if all(map(has_path(list(set(past_walking))),self.fitness.current_walked_tree)) and past_walking:
                break
            coverage = len(list(set(self.fitness.walked_tree)))/len(self.fitness.whole_tree)
            best_positions.update({f"{pos}": f"Cost is {cost} and coverage is {coverage}"})
            print(f"Real coverage is {coverage}")
            past_walking.extend(self.fitness.current_walked_tree)
        print(self.fitness.custom_weights)
        print(f"Positions and coverage are {best_positions}")
        print(f"The coverage of the matrix is {coverage}")
        print(f"whole tree is {list(set(self.fitness.whole_tree))} Walked tree:  {list(set(self.fitness.walked_tree))}")
        et = time.time()
        total_time = et - st
        print(f"Total elapsed time is {total_time} seconds")
        # print(f"Custom weights are {temp_arr}") 
    


