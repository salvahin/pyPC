import numpy as np
import pandas as pd
import csv
import os


### for Fitness
import pymoo.gradient.toolbox as anp

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE 
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES



from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

from pymoo.operators.sampling.lhs import LHS

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES



from test_fitness import TestGenerator
pop_size=10
SUT_dir = './SUT/'

def getSUTS():
    res = []
    # Iterate directory
    for file in os.listdir(SUT_dir):
        # check if current file is .py
        if file.endswith('.py'):
            res.append(file)
    return res

def getAlgorithm(alg):
    #This function receives a string representation of the algorithm and returns the configured pymoo algorithm object
    if alg == 'PSO':
        algorithm = PSO(pop_size=pop_size)
    elif alg == 'DE':
        algorithm = DE(pop_size=pop_size,variant='DE/best/1/bin')
    elif alg == 'PS':
        algorithm = PatternSearch()
    elif alg == 'GA':
        algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
    elif alg == 'NM':
        algorithm = NelderMead()
    elif alg == 'CMAES':
        algorithm = CMAES(x0=np.random.random(2))## This argument should change as he dimensions change
    elif alg == 'ES':
        algorithm = ES(n_offsprings=pop_size, rule=1.0 / 7.0)
    elif alg == 'SRES':
        algorithm = SRES(n_offsprings=pop_size, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)
    elif alg == 'ISRES':
        algorithm = ISRES(n_offsprings=pop_size, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)
    return algorithm
    """
    elif alg =='BRKGA':
        algorithm = BRKGA(n_elites=200, n_offsprings=700, n_mutants=100, bias=0.7, eliminate_duplicates=MyElementwiseDuplicateElimination())
    """





algs = ('PSO','DE','PS','GA','NM','CMAES','ES','SRES','ISRES')
data = []

SUTS = getSUTS()
for SUT in SUTS:
    print("*************\Analyzing ",SUT)
    for alg_name in algs:
        print("*************\nRunning ",alg_name)
        res = TestGenerator(SUT_dir+SUT,algorithm=getAlgorithm(alg_name),n_var=2)
        res.data['Alg'] = alg_name
        res.data['SUT'] = SUT
        data.append(res.data)
#print(data[4])

keys = data[0].keys()

with open('data.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    
