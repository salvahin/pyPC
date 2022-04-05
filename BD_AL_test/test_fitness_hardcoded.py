from scipy.spatial import distance
import numpy as np
from pyswarms.discrete.binary import BinaryPSO as binaryPSO 
from collections import deque, defaultdict
from ast2json import ast2json
import sys
import ast


al = [lambda a: lambda b: lambda c: (a>b) or (b<c), lambda b: lambda a: lambda c: (c+b)>(a+c)]
preds = ['( [0] > [1] ) or ( [1] < [2] )', ' ( [2] + [1] ) > ( [0] + [2] )']
k = 0.1

def fitness_function(param):
    """
    Fitness function combining both branch distance and approach level
    Must accept a (numpy.ndarray) with shape (n_particles, dimensions)
    Must return an array j of size (n_particles, )
    """
    particles_fitness = []
    for particle in param:
        sum_al = 0
        sum_bd = 0
        for index, pred in enumerate(preds):
            [pred:=pred.replace(f'[{index}]',f'{gene}') for index, gene in enumerate(particle)]
            print(pred)
            tokens = deque(pred.split())
            sum_bd += calc_expression(tokens)
            sum_al += approach_level(al[index], particle)

        normalized_bd = 1 - pow(1.001, -sum_bd)
        #print(normalized_bd)
        particles_fitness.append(normalized_bd+sum_al)
    return tuple(particles_fitness)


def calc_expression(tokens):
    """Calculate a list like [1, +, 2] or [1, +, (, 2, *, 3, )]"""

    lhs = tokens.popleft()

    if lhs == "(":
        lhs = calc_expression(tokens)

    else:
        lhs = int(lhs)

    operator = tokens.popleft()

    rhs = tokens.popleft()

    if rhs == "(":
        rhs = calc_expression(tokens)

    else:
        rhs = int(rhs)

    # We should be at the end of an expression, so there should be
    # either nothing left in the list, or just a closing parenthesis

    if tokens:
        assert tokens.popleft() == ")", \
            "bad expression, expected closing-paren"

    # Do the math
    # print(f"Doing the math for {lhs} {operator} {rhs}")

    if operator == "+" or operator == 'and':
        result = lhs + rhs

    elif operator == "-":
        result = lhs - rhs
    
    elif operator == '>':
        result = 0 if rhs - lhs == 0 else rhs - lhs + k
    
    elif operator == '<':
        result = 0 if lhs - rhs == 0 else lhs - rhs + k

    elif operator == "*":
        result = lhs * rhs

    elif operator == "/":
        result = lhs / rhs
    
    elif operator == 'or':
        result = min(lhs, rhs)
    
    elif operator == '==':
        result = 0 if lhs - rhs == 0 else abs(lhs - rhs) + k

    else:
        raise Exception("bad operator")

    return result

def approach_level(pred, value): 
    """
    Obtains the approach level of the branch to the ideal path
    """
    if pred(value):
        return 0
    return 1


if __name__ == '__main__':
    #shape (n_particles, n_dimensions)
    # a.shape[1] = number of values
    # print(visitor.function_names)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.3, 'k': 3, 'p': 2}
    bpso = binaryPSO(20, 3, options=options)
    cost, pos = bpso.optimize(fitness_function, iters=100)
    print(f"Best cost is {cost} and best position of particle is {pos}")
    