import numpy as np
import pandas as pd
import csv

from utils import get_dim, getSUTS
from test_fitness import TestGenerator



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



pop_size=10
SUT_dir = './SUT/'



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
        algorithm = CMAES(x0=np.random.random(2))## This argument should change as the dimensions change
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





algs = ('PSO','DE','PS','GA','NM','ES','SRES','ISRES')
data = []

SUTS = getSUTS(SUT_dir)
#SUTS = [ 'test.py']
for SUT in SUTS:
    SUT_path = SUT_dir+SUT
    func_dim,func_name = get_dim(SUT_path)
    #print("*************\Analyzing ",func_name, "dim: ",func_dim)
    for alg_name in algs:
        #print("*************\nRunning ",alg_name)
       
        res = TestGenerator(SUT_path,algorithm=getAlgorithm(alg_name),n_var=func_dim)
        res.data['Alg'] = alg_name
        res.data['SUT'] = func_name
        res.data['dim'] = func_dim
        data.append(res.data)
#print(data[4])

keys = data[0].keys()

with open('data.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    
