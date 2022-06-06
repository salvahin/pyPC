from multiprocessing.sharedctypes import Value
from tokenize import Double
import numpy as np
from itertools import product
import re
from functools import reduce
from pyswarms.discrete.binary import BinaryPSO as binaryPSO 
from pyswarms.single import GlobalBestPSO as GBPSO
from collections import deque, defaultdict
from ast2json import ast2json
import sys
import ast

k = 0.1
class Node:
    def __init__(self):
        self.statements = []
    def __str__(self) -> str:
        return f"{self.statements} \n"

class TreeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_names = []
        self.functions_trees = defaultdict(list)
        self.curr_func_json = None
        self.nodes = {}
        self.nodes['body'] = []
        self.if_nested_nodes = []
        self.if_nodes = {}
        self.while_nodes = {}
        self.curr_node = ''
        self._args = []
        self._vars_values = {}
        self.operators = {
            "And": "and",
            "Or": "or",
            "Gt": ">",
            "Lt": "<",
            "Eq": "==",
            "Add": "+",
            "Sub": "-",
            "Mult": "*",
            "Div": "/",
            "FloorDiv": "/",
            "Mod": "%",
            "Pow": "**",
            "NotEq": "!=",
            "LtE": "<=",
            "GtE": ">=",
            "In": "in",
            "NotIn": "not in",
            "Not": "not",
            "Invert": "~"
        }
        super().__init__()
    
    def _rename_while_keys(self, if_body="", else_if="", is_while=False, while_body="", if_parent=False):
        """
        Method to rename the while_node keys if is inside the if clause.
        Example: If x:
                    for y in range(1):
                         print(x)
        first it is going to parse the If then the for but the for keys parsed won't 
        have the If as a parent key so this will change for-test and for-body keys to 
        {if_body}-for-test and {if_body}-for-body keys
        """
        temp_dict = {}
        for key, value in self.while_nodes.items():
                temp_dict[f'{"if-"*if_body}{else_if}body-{key}'] = value
        # self.while_nodes = {}
        return temp_dict

    def _get_value_from_ast(self, obj):
        """
        Return the value of the ast object.
        """
        if obj['_type'] == 'Name':
            try:
                return f"self.{obj['id']}"
            except ValueError:
                return f"{obj['id']}"
        elif obj['_type'] == 'Num':
            return f"{obj['n']}"
        elif obj['_type'] == 'Str':
            return f"{obj['s']}"
        elif obj['_type'] == 'Constant':
            return f"{obj['value']}"
        elif obj['_type'] == 'Call':
            arguments = ','.join([self._parse_if_test(arg) for arg in obj['args']])
            print(f"{obj['func']['id']}({arguments})")
            return f"{obj['func']['id']}({arguments})"
        elif obj['_type'] == 'Break':
            return "pass"
        elif obj['_type'] == 'Continue':
            return "pass"
        elif isinstance(obj, ast.List):
            return [self._get_value_from_ast(e) for e in obj.elts]
        elif isinstance(obj, ast.Tuple):
            return tuple([self._get_value_from_ast(e) for e in obj.elts])
        # Probably passed a variable name.
        # Or passed a single word without wrapping it in quotes as an argument
        # ex: p.inflect("I plural(see)") instead of p.inflect("I plural('see')")
        raise NameError("name '%s' is not defined" % obj['_type']) 

    def re_structure_tree(self):
        """
        Re-structures the dictionary from the current function to a list of nodes
        that can be traversed on the metaheuristic algorithm.
        vg. nodes = [statement1, statement2, ... etc]
            arguments = ['z','x','y', ... etc]
        return: None.
        """
        #print(self.curr_func_json.keys())
        # print(self.curr_func_json['args'].keys())
        # print(self.curr_func_json['args']['defaults'])
        # print(self.curr_func_json['args']['kwarg'])
        # print(self.curr_func_json['args']['vararg'])
        # print(self.curr_func_json['args']['args'])
        #print(self.curr_func_json['body'])
        self._parse_args(self.curr_func_json['args']['args'])
        self._parse_body(self.curr_func_json['body'])

    def _parse_body(self, body):
        """
        
        """
        node = Node()
        node.statements.extend([f'self.{x} = [{num}]' for num, x in enumerate(self._args)])
        self.nodes['body'].append(node)
        print(f"INITIALIZING VARS {self.nodes['body'][0].statements}")
        for statement in body:
            # print(statement.keys())
            if statement['_type'] == 'If':
                self.nodes['body'].append(self._parse_if(statement))
            elif statement['_type'] == 'Expr':
                pass
            elif statement['_type'] == 'Assign':
                value = self._parse_if_test(statement['value'])
                statements = []
                for target in statement['targets']:
                    statements.append(f"{self._get_value_from_ast(target)} = {value}") 
                node = Node()
                node.statements.extend(statements)
                self.nodes['body'].append(node)
            elif statement['_type'] == 'AugAssign':
                value = self._parse_if_test(statement['value'])
                op = self.operators[statement['op']['_type']]
                statements = []
                statements.append(f"{self._get_value_from_ast(statement['target'])} {op}= {value}") 
                node.statements.extend(statements)
            elif statement['_type'] == 'BinOp':
                node = Node()
                node.statements.append(self._parse_if_test(statement))
                self.nodes['body'].append(node)
                
            elif statement['_type'] == 'Compare':
                node = Node()
                node.statements.append(self._parse_if_test(statement))
                self.nodes['body'].append(node)
            elif statement['_type'] == 'For':
                self.nodes['body'].append(self._parse_for(statement))

            elif statement['_type'] == 'While':
                self.nodes['body'].append(self._parse_while(statement))
            elif statement['_type'] == 'Break':
                node = Node()
                node.statements.append(self._parse_if_test(statement))
            elif statement['_type'] == 'Continue':
                node = Node()
                node.statements.append(self._parse_if_test(statement))


    def _parse_while(self, statement):
        """
        Parse the while body and condition repeat until condition is False
        """
        print(f"Begin While Statement {statement['_type']}")
        result = self._parse_if_test(statement['test']).strip()
        node = Node()
        node.statements.append(result)
        self.while_nodes['while-test'] = node
        # parse If body
        result = self._parse_if_body(statement['body'])
        # print(result)
        result = result if result.statements else Node()
        self.while_nodes['while-body'] = result

    def _parse_for(self, statement):
        """
        Parse the while body and condition repeat until condition is False
        """
        print(f"Begin For Statement {statement['_type']}")
        target = self._parse_if_test(statement['target'])
        iter = self._parse_if_test(statement['iter'])
        result = f"for {target} in " \
                 f"{iter}"
        node = Node()
        node.statements.append(result)
        for_node = {}
        for_node['for-test'] = node
        # parse If body
        result = self._parse_if_body(statement['body'])
        result = result if result.statements else Node()
        for_node['for-body'] = result
        return for_node


    def _parse_if(self, statement, is_else=False, else_if = ''):
        """
        Parse the if body and condition
        """
        print(f"BEGIN THE IF STATEMENT {statement['_type']} {else_if}")
        node = Node()
        # parse If test
        result = self._parse_if_test(statement['test']).strip()
        node.statements.append(result)
        if_node = {}
        if_node[f'if-{else_if}test'] = node
        result = self._parse_if_body(statement['body'])
        result = result if result.statements else Node()
        if_node[f'if-{else_if}body'] = result
        else_body = []
        for next_if in statement['orelse']:
            if next_if['_type'] == 'If':
                if_node.update(self._parse_if(next_if, is_else=True, else_if=f'{else_if}elif-'))
            else:
                else_body.append(next_if)
        if else_body:
            result = self._parse_if_body(else_body)
            if_node[f'if-{else_if if is_else else ""}else'] = result
        return if_node 

    
    def _parse_if_test(self, test, expression=""):
        if test['_type'] == 'BoolOp':
            op = self.operators[test['op']['_type']]
            # print(test['values'][0])
            left = self._parse_if_test(test['values'][0])
            right = self._parse_if_test(test['values'][1])
            expression += f" ( {left} {op} {right} )"
            return expression
        elif test['_type'] == 'BinOp':
            op = self.operators[test['op']['_type']]
            left = self._parse_if_test(test['left'])
            right = self._parse_if_test(test['right'])
            expression += f" ( {left} {op} {right} )"
            return expression
        elif test['_type'] == 'Compare':
            # print(test['left'])
            left = self._parse_if_test(test['left'])
            expression += f" ( {left}"
            for index, op in enumerate(test['ops']):
                op = self.operators[op['_type']]
                expression += f" {op}"
                right = self._parse_if_test(test['comparators'][index])
                expression += f" {right} )"
            return expression
        else:
            expression += f"{self._get_value_from_ast(test)}"
            return expression

    def _parse_if_body(self, body) -> Node:
        """
        
        """
        node = Node()
        for statement in body:
            if statement['_type'] == 'If':              
                node.statements.append(self._parse_if(statement))

            elif statement['_type'] == 'Expr':
                pass
            elif statement['_type'] == 'Assign':
                value = self._parse_if_test(statement['value'])
                statements = []
                for target in statement['targets']:
                    statements.append(f"{self._get_value_from_ast(target)} = {value}") 
                node.statements.extend(statements)
            elif statement['_type'] == 'AugAssign':
                value = self._parse_if_test(statement['value'])
                op = self.operators[statement['op']['_type']]
                statements = []
                statements.append(f"{self._get_value_from_ast(statement['target'])} {op}= {value}") 
                node.statements.extend(statements)
            elif statement['_type'] == 'BinOp':
                node.statements.append(self._parse_if_test(statement))
                
            elif statement['_type'] == 'Compare':
                node.statements.append(self._parse_if_test(statement))
            elif statement['_type'] == 'BoolOp':
                node.statements.append(self._parse_if_test(statement))
            elif statement['_type'] == 'For':
                node.statements.append(self._parse_for(statement))
            elif statement['_type'] == 'While':
                node.statements.append(self._parse_while(statement))
            elif statement['_type'] == 'Break':
                node.statements.append(self._parse_if_test(statement))
            elif statement['_type'] == 'Continue':
                node.statements.append(self._parse_if_test(statement))
        return node

    def _parse_args(self, args):
        """
        Extracts from the dictionary the argument names
        args: list of arguments
        list of argument names eg. ['x','y','a'].
        example of use args.index('x'), index is used to obtain the position of each input.
        return: None
        """
        for arg in args:
            self._args.append(arg['arg'])

    def visit_FunctionDef(self, node):
        """
        Default visitor function where the tree of each function is parsed into json.
        node: NodeVisitor
        return: None
        """
        # print(f"Function Definition {node._fields}")
        self.function_names.append(node.name)
        self.functions_trees[node.name] = {}
        result = ast2json(node)
        print(result)
        self.curr_func_json = result
        self.re_structure_tree()
        
    def visit_Import(self, node):
        result = ast2json(node)
        print(result)
        node2 = Node()
        node2.statements.append(f"import {','.join([name['name'] for name in result['names']])}")
        for name in result['names']:
                node2.statements.append(f"self.{name['name']} = {name['name']}")
        self.nodes['body'].append(node2)
        ast.NodeVisitor.generic_visit(self, node)
    def visit_ImportFrom(self, node):
        result = ast2json(node)
        print(result)
        node2 = Node()
        if list(filter(lambda x: True if x else False, [name['asname'] for name in result['names']])):
            node2.statements.append(f"from {result['module']} import " \
                             f"{','.join([name['name'] for name in result['names']])} as " \
                             f"{','.join([name['asname'] for name in result['names']])}")
            self.nodes['body'].append(node2)
        else:
            node2.statements.append(f"from {result['module']} import " \
                             f"{','.join([name['name'] for name in result['names']])}")
            for name in result['names']:
                node2.statements.append(f"self.{name['name']} = {name['name']}")
            self.nodes['body'].append(node2)
        ast.NodeVisitor.generic_visit(self, node)
    # def visit_BoolOp(self, node):
    #     print(f"Bool op {node._fields}")
    #     print(f"{node.op}  {node.values}")
    #     ast.NodeVisitor.generic_visit(self, node)

    # def visit_Str(self, node):
    #     print(f"Node type: Str\nFields: {node._fields}")
    #     print(f"{node.s}")
    #     ast.NodeVisitor.generic_visit(self, node)
    # def visit_Num(self, node):
    #     print(f"Node type: Num\nFields: {node._fields}")
    #     print(f"{node.n}")
    #     ast.NodeVisitor.generic_visit(self, node)              
    # def visit_Expr(self, node):
    #     print(f"Node type: Expr\nFields: {node._fields}")
    #     ast.NodeVisitor.generic_visit(self, node)
    # def visit_BinOp(self, node):
    #     print(f"Node type: Expr\nFields: {node._fields}")
    #     ast.NodeVisitor.generic_visit(self, node)
    # def visit_Name(self, node):
    #     print('Node type: Name\nFields:', node._fields)
    #     print(f"{node.id}")
    #     ast.NodeVisitor.generic_visit(self, node)

    # def visit_Constant(self, node):
    #     print('Node type: Constant\nFields:', node._fields)
    #     ast.NodeVisitor.generic_visit(self, node)

    # def visit_Pass(self, node):
    #     print('Node type: Pass\nFields:', node._fields)
    #     ast.NodeVisitor.generic_visit(self, node)
class Fitness:

    def measure_coverage(self):
        nodes = visitor.nodes['body']

    def fitness_function(self, param):
        """
        Fitness function combining both branch distance and approach level
        Must accept a (numpy.ndarray) with shape (n_particles, dimensions)
        Must return an array j of size (n_particles, )
        """
        particles_fitness = []
        for particle in param:
            sum_al = 0
            sum_bd = 0
            for node in visitor.nodes['body']:
                if isinstance(node, dict):
                    sum_al, sum_bd = self.resolve_if(node, particle, sum_al, sum_bd)
                        # ap = approach_level(p,p)
                    #if key == ''
                else:
                    for statement in node.statements:
                        print(statement)
                        [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                        exec(statement)
            normalized_bd = 1 + (-1.001 ** -abs(sum_bd))
            print(normalized_bd)
            particles_fitness.append(float(normalized_bd+sum_al))
        return tuple(particles_fitness)

    def resolve_if(self, node, particle, sum_al, sum_bd, al=1):
        enters_if = False
        for key, _ in node.items():
            if 'body' in key:
                continue
            if 'while' in key:
                statement = node[key].statements[0]
                while exec(statement):
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    sum_aplevel = sum_brd = 0
                    for statement in statements:
                        if isinstance(statement, dict):
                            sum_aplevel, sum_brd = self.resolve_if(statement, particle, sum_al, sum_bd)
                            print(f"returning cost value inside for {sum_aplevel_temp}")
                            sum_aplevel += sum_aplevel_temp
                            sum_brd += sum_brd_temp
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
                    for statement in statements:
                        if isinstance(statement, dict):
                            print(f"evaluating statement inside {key}  {statement}")
                            sum_aplevel_temp, sum_brd_temp = self.resolve_if(statement, particle, sum_al, sum_bd)
                            print(f"returning cost value inside for {sum_aplevel_temp}")
                            sum_aplevel += sum_aplevel_temp
                            sum_brd += sum_brd_temp

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
                break
            if not enters_if and 'elif' in key:
                statement = node[key].statements[0]
                tokens = deque(statement.split())
                sum_bd += self.calc_expression(tokens)
                al = self.approach_level(statement)
                sum_al += al
                if not al:
                    print(f"Enters ElIF body {key}")
                    enters_if = True
                    statements = node[f'{key}-body'].statements
                    sum_aplevel = sum_brd = 0
                    for statement in statements:
                        if isinstance(statement, dict):
                            sum_aplevel, sum_brd = self.resolve_if(statement, particle, sum_al, sum_bd)
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
                sum_aplevel = sum_brd = 0
                for statement in statements:
                    if isinstance(statement, dict):
                        sum_aplevel, sum_brd = self.resolve_if(statement, particle, sum_al, sum_bd)
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
                sum_bd += self.calc_expression(tokens)
                al = self.approach_level(statement)
                sum_al += al
                if not al:
                    print(f"Enters IF body {key}")
                    enters_if = True
                    statements = node[f'{key.replace("-test", "-body")}'].statements
                    sum_aplevel = sum_brd = 0
                    for statement in statements:
                        if isinstance(statement, dict):
                            sum_aplevel, sum_brd = self.resolve_if(statement, particle, sum_al, sum_bd)
                        else:
                            [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                            try:
                                exec(statement)
                            except NameError as e:
                                name = str(e).split()[1].replace("'", "")
                                exec(statement.replace(name, f'self.{name}'))
                        
        return sum_al, sum_bd
    
    
    def calc_expression(self, tokens):
        """Calculate a list like [1, +, 2] or [1, +, (, 2, *, 3, )]"""
        lhs = tokens.popleft()
    
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
    
    
if __name__ == '__main__':
    #shape (n_particles, n_dimensions)
    # a.shape[1] = number of values
    #with open("test.py", 'r+') as filename:
    #   lines = filename.readlines()
    #   tree = ast.parse(''.join(lines))
    with open("test.py", 'r+') as filename:
       lines = filename.readlines()
       tree = ast.parse(''.join(lines))
    # print(ast.dump(tree))
    tree = ast.parse(tree)
    visitor = TreeVisitor()
    visitor.visit(tree)
    print(visitor.nodes)
    
    # print(visitor.function_names)
    #options = {'c1': 0.9, 'c2': 0.5, 'w': 0.9, 'k': 3, 'p': 3}
    # bpso = binaryPSO(20, 2, options=options)
    #gbpso = GBPSO(20,2,options=options)
    #fitness = Fitness()
    #fitness.calc_expression(deque('( (  ( 1 == 3 ) and  ( 101 > 100 ) ) or  ( 100 + 3 ) )'.split()))
    #cost, pos = gbpso.optimize(fitness.fitness_function, iters=100)
    #print(f"Best cost is {cost} and best position of particle is {pos}")
    