from lib2to3.pytree import convert
from pyswarms.single import GlobalBestPSO as GBPSO
from pyswarms.single import LocalBestPSO as LBPSO
import time
import ast
import numpy as np
import pandas as pd
from tree_converter import TreeVisitor
from test_fitness import Fitness
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

paths = {
    "test_programs/test_game_programs/function_only_testings/rock_paper_scissor_player_choice.py": 1,
    "test_programs/test_game_programs/function_only_testings/bounce_draw.py": 2,
    "test_programs/test_game_programs/function_only_testings/guess_the_number_input_guess.py": 1,
    "test_programs/test_game_programs/function_only_testings/jogo_da_velha_python_actualizar_jogadas.py": 1,
    "test_programs/test_game_programs/function_only_testings/rock_paper_scissor_number_to_name.py": 1,
    "test_programs/test_game_programs/function_only_testings/TRPG_character_create_character.py": 1,
    "test_programs/bubble_sort.py": 4,
    "test_programs/minimum.py": 4,
    "test_programs/three_number_sort.py": 3,
    "test_programs/trig_area.py": 3
}

algorithms = [PSO(pop_size=100, w= 0.7, c1=2, c2=2),
              GA(pop_size=100, eliminate_duplicates=True),
              DE(
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.3,
                 dither="vector",
                 jitter=False
                ),
              NelderMead(),
              PatternSearch(),
              SRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2),
              ISRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)]
# algorithm = CMAES(x0=np.random.random(dimensions))  # Only for more than 1 dimension



def run_algorithm(algorithm, problem=None):
    """
    Function that runs the selected algorithm into the tree nodes
    """
    try:
        cost, pos = algorithm.optimize(fitness.fitness_function, iters=100)
    except:
        result = minimize(
                          problem, 
                          algorithm, 
                          ('n_iter', 100),
                          
                          verbose=True)
    return result.F, result.X

def convert_tree(path):
    """
    Function to convert the selected function file into tree traversable nodes
    """
    with open(path, 'r+') as filename:
       lines = filename.readlines()
       tree = ast.parse(''.join(lines))
    # print(ast.dump(tree))
    tree = ast.parse(tree)
    visitor = TreeVisitor()
    visitor.visit(tree)
    print(visitor.nodes)
    return visitor

class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_cost = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))

class FitnessProblem(Problem):

    def __init__(self, fitness=None, dimensions=None):
        xl = -999999 # longest number in an int

        xu = 999999 # longest number in an int
        self.fitness = fitness
        self.ndim = dimensions
        super().__init__(n_var=dimensions, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fitness.fitness_function(x)


if __name__ == '__main__':
    # 30 executions per algorithm
    results_df = pd.DataFrame()
    algorithms_list = []
    programs = []
    coverages = []
    times = []
    iteration = []
    for path, dimensions in paths.items():
    #path = "test_programs/test_game_programs/function_only_testings/bounce_draw.py"
        visitor = convert_tree(path)
        fitness = Fitness(visitor)
        #options = {'c1': 2, 'c2': 2, 'w': 0.7}
        # particles = 100
        # dimensions = 2
        problem = FitnessProblem(fitness, dimensions)
        #if dimensions > 1:
        #    algorithms.append(CMAES(x0=np.random.random(dimensions)))
        for algorithm in algorithms:
            for i in range(1, 31):
                max_paths = 0
                coverage = 0
                past_walking = []
                best_positions = {}
                more_paths = True
                fitness.current_walked_tree = []
                fitness.walked_tree = []
                iteration.append(i)
                st = time.time()
                programs.append(path.split('/')[-1])
                algorithms_list.append(algorithm.__module__)
                # algorithm = algorithms[1]
                # algorithm = CMAES(x0=np.random.random(dimensions))  # Only for more than 1 dimension
                # gbpso = GBPSO(particles,dimensions,options=options)
                while more_paths:
                    #cost, pos = run_algorithm(gbpso)
                    cost, pos = run_algorithm(algorithm, problem)
                    particle_pos = np.array([pos],np.float32)
                    fitness.resolve_path(particle_pos)
                    has_path = lambda x: lambda y: y in x
                    if all(map(has_path(list(set(past_walking))),fitness.current_walked_tree)) and past_walking:
                        break
                    coverage = len(list(set(fitness.walked_tree)))/len(fitness.whole_tree)
                    best_positions.update({f"{pos}": f"Cost is {cost} and coverage is {coverage}"})
                    print(f"Real coverage is {coverage}")
                    past_walking.extend(fitness.current_walked_tree)
                    # plot_cost_history(cost_history=gbpso.cost_history)
                    # plt.show()
                print(fitness.custom_weights)
                print(f"Positions and coverage are {best_positions}")
                print(f"The coverage of the matrix is {coverage*100}%")
                print(f"whole tree is {list(set(fitness.whole_tree))} Walked tree:  {list(set(fitness.walked_tree))}")
                et = time.time()
                total_time = et - st
                times.append(total_time)
                coverages.append(f"{coverage*100}%")
                print(f"Total elapsed time is {total_time} seconds")
    results_df['Algorithm'] = algorithms_list
    results_df['Iteration'] = iteration
    results_df["Code Tested"] = programs
    results_df['Coverage'] = coverages
    results_df['Execution Time'] = times
    results_df.to_csv("output_results.csv", index=False)
            # print(f"Custom weights are {temp_arr}")    