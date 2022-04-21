from multiprocessing.sharedctypes import Value
from scipy.spatial import distance
import numpy as np
import re
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
        self.if_nodes = {}
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
                self._parse_if(statement)
                self.nodes['body'].append(self.if_nodes)
                self.if_nodes = {}
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
                pass
            elif statement['_type'] == 'While':
                pass


    def _parse_if(self, statement, is_else=False, if_body="", else_if = ''):
        """
        Parse the if body and condition
        """
        print(f"BEGIN THE IF STATEMENT {statement['_type']}")
        node = Node()
        # parse If test
        result = self._parse_if_test(statement['test']).strip()

        node.statements.append(result)
        if is_else:
            self.if_nodes[f'{if_body}if-{else_if}test'] = node
        else:
            self.if_nodes[f'{if_body}if-test'] = node
        # parse If body
        result = self._parse_if_body(statement['body'], if_body)
        # print(result)
        result = result if result.statements else Node()
        if is_else:
            self.if_nodes[f'{if_body}if-{else_if}body'] = result
        else:
            self.if_nodes[f'{if_body}if-body'] = result
        else_body = []
        for next_if in statement['orelse']:
            if next_if['_type'] == 'If':
                self._parse_if(next_if, is_else=True, if_body=if_body, else_if=f'{else_if}elif-')
            else:
                else_body.append(next_if)
        if else_body:
            result = self._parse_if_body(else_body, if_body)
            self.if_nodes[f'{if_body}if-{else_if if is_else else ""}else'] = result

    
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

    def _parse_if_body(self, body, if_body="") -> Node:
        """
        
        """
        node = Node()
        for statement in body:
            if statement['_type'] == 'If':
                if_body += 'if-'
                print("NESTED IF")
                self._parse_if(statement, if_body=if_body)
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
        # print(result)
        self.curr_func_json = result
        self.re_structure_tree()
        
    def visit_Import(self, node):
        result = ast2json(node)
        print(result)
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
                    for statement in statements:
                        [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                        try:
                            exec(statement)
                        except NameError as e:
                            name = str(e).split()[1].replace("'", "")
                            exec(statement.replace(name, f'self.{name}'))
            elif not enters_if and 'else' in key:
                print(f"Enters ELSE body {key}")
                statements = node[key].statements
                for statement in statements:
                    [statement:=statement.replace(f'[{index}]', f'{gene}') for index, gene in enumerate(particle)]
                    try:
                        exec(statement)
                    except NameError as e:
                            name = str(e).split()[1].replace("'", "")
                            exec(statement.replace(name, f'self.{name}'))
            elif 'else' not in key and 'elif' not in key:
                statement = node[key].statements[0]
                tokens = deque(statement.split())
                print(f"ENTERS IF {key}")
                sum_bd += self.calc_expression(tokens)
                al = self.approach_level(statement)
                sum_al += al
                if not al:
                    print(f"Enters IF body {key}")
                    enters_if = True
                    statements = node[f'{key.replace("-test", "")}-body'].statements
                    for statement in statements:
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
    with open("trig_area.py", 'r+') as filename:
       lines = filename.readlines()
       tree = ast.parse(''.join(lines))
    print(ast.dump(tree))
    tree = ast.parse(tree)
    visitor = TreeVisitor()
    visitor.visit(tree)
    print(visitor.nodes)
    
    # print(visitor.function_names)
    options = {'c1': 0.9, 'c2': 0.5, 'w': 0.9, 'k': 3, 'p': 3}
    # bpso = binaryPSO(20, 2, options=options)
    gbpso = GBPSO(20,3,options=options)
    fitness = Fitness()
    #fitness.calc_expression(deque('( (  ( 1 == 3 ) and  ( 101 > 100 ) ) or  ( 100 + 3 ) )'.split()))
    cost, pos = gbpso.optimize(fitness.fitness_function, iters=100)
    print(f"Best cost is {cost} and best position of particle is {pos}")
    