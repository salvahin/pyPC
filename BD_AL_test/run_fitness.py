from lib2to3.pytree import convert
from pyswarms.single import GlobalBestPSO as GBPSO
from pyswarms.single import LocalBestPSO as LBPSO
import time
import ast
import numpy as np
from tree_converter import TreeVisitor
from test_fitness import Fitness

def run_algorithm(algorithm):
    """
    Function that runs the selected algorithm into the tree nodes
    """
    cost, pos = algorithm.optimize(fitness.fitness_function, iters=100)
    return cost, pos

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


if __name__ == '__main__':
    st = time.time()
    path = "test_programs/test_game_programs/function_only_testings/rock_paper_scissor_player_choice.py"
    visitor = convert_tree(path)
    best_positions = {}
    fitness = Fitness(visitor)
    more_paths = True
    max_paths = 0
    coverage = 0
    past_walking = []
    options = {'c1': 2, 'c2': 2, 'w': 0.7}
    particles = 100
    dimensions = 1
    gbpso = GBPSO(particles,dimensions,options=options)
    while more_paths:
        
        cost, pos = run_algorithm(gbpso)
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
    print(f"Total elapsed time is {total_time} seconds")
    # print(f"Custom weights are {temp_arr}")    