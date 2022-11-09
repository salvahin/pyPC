### for Fitness
import pymoo.gradient.toolbox as anp

from pymoo.algorithms.soo.nonconvex.de import DE 
from pymoo.operators.sampling.lhs import LHS

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES



from test_fitness import TestGenerator
pop_size=100

def getAlgorithm(alg):
    #This function receives a string representation of the algorithm and returns the configured pymoo algorithm object
    if alg == 'PSO':
        algorithm = PSO(pop_size=pop_size)
    elif alg == 'DE':
        algorithm = DE(pop_size=pop_size,variant='DE/best/1/bin')
    elif alg == 'PS':
        algorithm = PatternSearch()
    return algorithm
    

TestGenerator("./SUT/test.py",algorithm=getAlgorithm('DE'),n_var=2)