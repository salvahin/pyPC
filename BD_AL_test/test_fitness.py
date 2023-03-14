from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from IPython.display import Image
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from tokenize import Double
import numpy as np
from itertools import product
import re
import math
from functools import reduce
from pyparsing import nested_expr


k = 0.1
class Fitness:

    def __init__(self, visitor) -> None:
        self.complete_coverage = {}
        self.coverage = 0
        self.walked_tree = []
        self.current_walked_tree = []
        self.whole_tree = set()
        self.custom_weights = {}
        self.visitor = visitor

    def explore(self, node):
        result = 0
        for x, v in node.items():
            print(f"finding if in {x}")
            if 'if' in x:
                return 1
            for y in v.statements:
                if isinstance(y, dict):
                    result = self.explore(y)
        return result
    def round_half_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n*multiplier + 0.5) / multiplier
    def fitness_function(self, param):
        """
        Fitness function combining both branch distance and approach level
        Must accept a (numpy.ndarray) with shape (n_particles, dimensions)
        Must return an array j of size (n_particles, )
        """
        particles_fitness = []
        for index_par, particle in enumerate(param):
            sum_al = 0
            sum_bd = 0
            ifs_num = 0
            self.coverage = 0
            for pos, node in enumerate(self.visitor.nodes['body']):
                if isinstance(node, dict):
                    sum_al, sum_bd, coverage, if_num = self.resolve_if(node, particle, sum_al, sum_bd, if_num=0, pos=pos)
                    #if 'if' in list(node.keys())[0]:
                    #    ifs_num += 1
                    ifs_num += self.explore(node)
                    print(f"COVERAGE and values are {coverage} {sum_al} {sum_bd} {if_num}")
                    self.coverage += coverage if if_num == 0 else (coverage/if_num)
                else:
                    for statement in node.statements:
                        if not 'import' in statement:
                            [statement:=statement.replace(f'[{index}]', f'{self.round_half_up(gene, 1)}') for index, gene in enumerate(particle)
                            if not re.match(r'\b([a-zA-Z_.0-9]*)\[[0-9]+\]', statement.split('=')[1].strip())]
                        exec(statement)

            normalized_bd = 1 + (-1.001 ** -abs(sum_bd))
            print(normalized_bd)
            self.coverage = 1 if self.coverage == 0 else self.coverage
            complete_execution_coverage = (self.coverage/ifs_num) if ifs_num > 0 else 1
            self.complete_coverage.update({f"{float(normalized_bd+sum_al)}": complete_execution_coverage})
            particles_fitness.insert(index_par, float(normalized_bd+sum_al))
        return np.array(particles_fitness)

    def resolve_if(self, node, particle, sum_al, sum_bd, al=1, if_num=0, pos=0, nested=0, count=''):
        enters_if = False
        coverage = 0
        len_statements2 = len(list(filter(lambda x: True if 'test' in x[0] else False, node.items())))
        aux_count = -1
        temp = ''
        for key, _ in node.items():
            if 'body' in key:
                continue
            aux_count += 1
            if count:
                temp = count
                count = f"{count}-{nested}.{aux_count}"
            else:
                count = f"{pos}-{nested}.{aux_count}"
            self.whole_tree.add(f'{count}')
            if 'while' in key:
                statement = node[key].statements[0]
                len_statements = 0
                while exec(statement):
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    len_statements += 1
                    sum_aplevel = sum_brd = 0
                    for ind, statement1 in enumerate(statements):
                        if isinstance(statement1, dict):
                            sum_aplevel, sum_brd_temp, coverage2, if_num = self.resolve_if(statement1, particle, sum_al, sum_bd, 1, if_num, pos=pos, nested=ind, count=count)
                            print(f"nested returned on for{if_num}")
                            sum_aplevel += sum_aplevel_temp
                            sum_brd += sum_brd_temp
                            enters_if = False
                            # coverage += coverage2
                        else:
                            [statement1:=statement1.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement1.replace(name, f'self.{name}'))
                print(f"moduling cost {sum_aplevel} to a value between 1 and 0, iter is {module_cost} {sum_aplevel/module_cost}")
                sum_al += sum_aplevel/module_cost
                sum_bd += sum_brd/module_cost
                # coverage = coverage/len_statements
                break
            if 'for' in key and not 'else' in key and not 'elif' in key and not 'if' in key:
                print(f"Enters {key}")
                test = node[key].statements[0]
                iters = test.split('in')[0][4:].strip().split(',')
                sum_aplevel = sum_brd = 0
                module_cost = len([x for x in product(eval(test.split('in')[1]))])
                print(f"for test to iterate is {test.split('in')[1]}")
                for x in product(eval(test.split('in')[1])):
                    if len(iters) == 1:
                        exec(f'{iters[0]} = x[0]')
                    else:
                        for num, iter in enumerate(iters):
                            exec(f'{iter} = x[{num}]')
                    statements = node[f"{key.replace('-test', '-body')}"].statements
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            print(f"evaluating statement inside {key}  {statement}")
                            sum_aplevel_temp, sum_brd_temp, coverage2, if_num = self.resolve_if(statement, particle, sum_al, sum_bd, 1, if_num, pos=pos, nested=ind, count=count)
                            sum_aplevel += sum_aplevel_temp
                            coverage += coverage2
                            sum_brd += sum_brd_temp
                            enters_if = False

                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
                print(f"moduling cost {sum_aplevel} to a value between 1 and 0, iter is {module_cost} {sum_aplevel/module_cost}")
                sum_al += sum_aplevel/module_cost
                sum_bd += sum_brd/module_cost
                print(f"Coverage cost is {coverage}   {if_num}")
                #coverage = (coverage/(module_cost)) if module_cost > 0 else coverage
                break
            if not enters_if and 'elif' in key and 'else' not in key:
                if_num += 1
                statement = node[key].statements[0]
                tokens = deque(statement.split())
                al = self.approach_level(statement)
                bd = self.calc_expression(tokens)
                sum_bd += bd
                sum_al += al
                if not al:
                    print(f"Enters ElIF body {key}")
                    if count in self.custom_weights.keys():
                        sum_bd += self.custom_weights[count]
                        sum_al += self.custom_weights[count]
                    coverage += 1
                    enters_if = True
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    sum_aplevel = sum_brd = 0
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            sum_aplevel, sum_brd, coverage2, if_num = self.resolve_if(statement, particle, sum_al, sum_bd, 1, if_num, pos=pos, nested=ind, count=count)
                            coverage += coverage2
                            sum_bd += sum_brd
                            sum_al += sum_aplevel
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}')) 
            elif not enters_if and 'else' in key:
                print(f"Enters ELSE body {key}")
                if_num += 1
                statements = node[key].statements
                len_statements = len(statements)
                sum_aplevel = sum_brd = 0
                coverage += 1
                if count in self.custom_weights.keys():
                    sum_bd += self.custom_weights[count]
                    sum_al += self.custom_weights[count]
                for ind, statement in enumerate(statements):
                    if isinstance(statement, dict):
                        sum_aplevel, sum_brd, coverage2, if_num = self.resolve_if(statement, particle, sum_al, sum_bd, 1, if_num, pos=pos, nested=ind, count=count)
                        coverage += coverage2
                        sum_bd += sum_brd
                        sum_al += sum_aplevel
                    else:
                        [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                        try:
                            exec(statement)
                        except NameError as e:
                            name = str(e).split()[1].replace("'", "")
                            exec(statement.replace(name, f'self.{name}'))
            elif 'else' not in key and 'elif' not in key:
                statement = node[key].statements[0]
                if_num += 1 
                tokens = deque(statement.split())
                print(f"ENTERS IF {key} {tokens}")
                al = self.approach_level(statement)
                bd = self.calc_expression(tokens)
                sum_bd += bd
                sum_al += al
                if not al:
                    print(f"Enters IF body {key}")
                    if count != '2-0.0-0.0-0.0':
                        print("hello")
                    if count in self.custom_weights.keys():
                        sum_bd += self.custom_weights[count]
                        sum_al += self.custom_weights[count]
                    coverage += 1
                    enters_if = True
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    len_statements = len(statements)
                    sum_aplevel = sum_brd = 0
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            sum_aplevel, sum_brd, coverage2, if_num = self.resolve_if(statement, particle, sum_al, sum_bd, 1, if_num, pos=pos, nested=ind, count=count)
                            coverage += coverage2
                            sum_bd += sum_brd
                            sum_al += sum_aplevel
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
            count = '' if not temp else temp
        coverage = (coverage/len_statements2) if len_statements2 > 0 else 1                
        return sum_al, sum_bd, coverage, if_num
    
    
    def calc_expression(self, tokens):
        """Calculate a list like [1, +, 2] or [1, +, (, 2, *, 3, )]"""
        lhs = tokens.popleft()
        if lhs == 'not':
            print("skipping the math since unary operator is exception")
            return 0
        if lhs == "(":
            lhs = self.calc_expression(tokens)
    
        else:
            try:
                print(f"Value trying to parse {lhs}")
                lhs = int(lhs)
            except ValueError:
                lhs = eval(lhs)
    
        if not tokens:
            return lhs
        operator = tokens.popleft()
    
        rhs = tokens.popleft()
    
        if rhs == "(":
            rhs = self.calc_expression(tokens)
    
        else:
            try:
                rhs = int(rhs)
            except ValueError:
                rhs = eval(rhs)
    
        # We should be at the end of an expression, so there should be
        # either nothing left in the list, or just a closing parenthesis
    
        if tokens:
            assert tokens.popleft() == ")", \
                "bad expression, expected closing-paren"
    
        # Do the math
        print(f"Doing the math for {lhs} {operator} {rhs}")
    
        if operator == "+" or operator == 'and':
            result = lhs + rhs
    
        elif operator == "-":
            result = lhs - rhs
        
        elif operator == '>':
            result = 0 if rhs - lhs < 0 else rhs - lhs + k
        
        elif operator == '<':
            result = 0 if lhs - rhs < 0 else lhs - rhs + k

        elif operator == '>=':
            result = 0 if rhs - lhs <= 0 else rhs - lhs + k

        elif operator == '<=':
            result = 0 if lhs - rhs <= 0 else lhs - rhs + k
    
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
    
    def approach_level(self, pred): 
        """
        Obtains the approach level of the branch to the ideal path
        """
        print(f"EVAL {eval(pred)} ")
        if eval(pred):
            return 0
        return 1
    
    def resolve_path(self, param, costs=[1048576]):
        """
        Function that travels through the nodes with the given solution and it returns
        the travelled nodes in structure like 1-2-1-1.1 that means that each number
        represents a branch in order of appearence from top to bottom and the decimal
        part represents nested branches also numbered as they appear.

        Must accept a (numpy.ndarray) with shape (n_particles, dimensions)
        Must return an array j of size (n_particles, )
        """
        for index_par, particle in enumerate(param):
            self.current_walked_tree = []
            for index, node in enumerate(self.visitor.nodes['body']):
                if isinstance(node, dict):
                    self.resolve_dict_path(node, particle, f'{index}')
                else:
                    for statement in node.statements:
                        if not 'import' in statement:
                            [statement:=statement.replace(f'[{index}]', f'{self.round_half_up(gene, 1)}') for index, gene in enumerate(particle)
                            if not re.match(r'\b([a-zA-Z_.0-9]*)\[[0-9]+\]', statement.split('=')[1].strip())]
                        exec(statement)
            prefix_path = []
            has_element = lambda x: lambda y: x in y
            for x in sorted(self.current_walked_tree, key=len, reverse=True):
                if not any(map(has_element(x), prefix_path)):
                    prefix_path.append(x)
            for x in prefix_path:
                divide_ = x.count('-')
                self.custom_weights.update({f"{x}": max(costs)/2**divide_})
            self.current_walked_tree = prefix_path

    def resolve_dict_path(self, node, particle, pos, al=1, nested=0, count=''):
        enters_if = False
        aux_count = -1
        temp = ''
        print(node.items())
        for key, _ in node.items():
            if 'body' in key:
                continue
            aux_count += 1
            if count:
                temp = count
                count = f"{count}-{nested}.{aux_count}"
            else:
                count = f"{pos}-{nested}.{aux_count}"
            if 'while' in key:
                statement = node[key].statements[0]
                self.walked_tree.append(f'{count}')
                self.current_walked_tree.append(f'{count}')
                while exec(statement):
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            self.resolve_dict_path(statement, particle, pos, 1, ind, count)
                            enters_if = False
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
                break
            if 'for' in key and not 'else' in key and not 'elif' in key and not 'if' in key:
                print(f"Enters {key}")
                test = node[key].statements[0]
                iters = test.split('in')[0][4:].strip().split(',')
                self.walked_tree.append(f'{count}')
                self.current_walked_tree.append(f'{count}')
                for x in product(eval(test.split('in')[1])):
                    if len(iters) == 1:
                        exec(f'{iters[0]} = x[0]')
                    else:
                        for num, iter in enumerate(iters):
                            exec(f'{iter} = x[{num}]')
                    statements = node[f"{key.replace('-test', '-body')}"].statements
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            print(f"evaluating statement inside {key}  {statement}")
                            self.resolve_dict_path(statement, particle, pos, 1, ind, count)
                            enters_if = False
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
                break
            if not enters_if and 'elif' in key and 'else' not in key:
                statement = node[key].statements[0]
                tokens = deque(statement.split())
                al = self.approach_level(statement)
                if not al:
                    print(f"Enters ElIF body {key}")
                    enters_if = True
                    self.walked_tree.append(f'{count}')
                    self.current_walked_tree.append(f'{count}')
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            self.resolve_dict_path(statement, particle, pos, 1, ind, count)
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
            elif not enters_if and 'else' in key:
                print(f"Enters ELSE body {key}")
                statements = node[key].statements
                self.walked_tree.append(f'{count}')
                self.current_walked_tree.append(f'{count}')
                for ind, statement in enumerate(statements):
                    if isinstance(statement, dict):
                        self.resolve_dict_path(statement, particle, pos, 1, ind, count)
                    else:
                        [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                        try:
                            exec(statement)
                        except NameError as e:
                            name = str(e).split()[1].replace("'", "")
                            exec(statement.replace(name, f'self.{name}'))
            elif 'else' not in key and 'elif' not in key:
                statement = node[key].statements[0]
                tokens = deque(statement.split())
                print(f"ENTERS IF {key} {tokens}")
                al = self.approach_level(statement)
                if not al:
                    self.walked_tree.append(f'{count}')
                    self.current_walked_tree.append(f'{count}')
                    print(f"Enters IF body {key}")
                    enters_if = True
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    for ind, statement in enumerate(statements):
                        if isinstance(statement, dict):
                            self.resolve_dict_path(statement, particle, pos, 1, ind, count)
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))    
            count = '' if not temp else temp          
