from collections import deque, defaultdict
from ast2json import ast2json
import ast

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
            "Invert": "~",
            "USub": "-"
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
            value = ""
            try:
                value = f"\"{obj['value']}\"" if obj['value'].isalpha() else obj['value']
            except AttributeError:
                value = obj['value']
            return f"{value}"
        elif obj['_type'] == 'Call':
            arguments = ','.join([self._parse_if_test(arg) for arg in obj['args']])
            print(f"{obj['func']['id']}({arguments})")
            return f"{obj['func']['id']}({arguments})"
        elif obj['_type'] == 'Break':
            return "pass"
        elif obj['_type'] == 'Continue':
            return "pass"
        elif obj['_type'] == 'Tuple':
            return ','.join([self._get_value_from_ast(element) for element in obj['elts']])
        elif obj['_type'] == 'List':
            return f"[{','.join([self._get_value_from_ast(e) for e in obj['elts']])}]"
        elif obj['_type'] == 'Subscript':
            return f"{self._get_value_from_ast(obj['value'])}[{''.join(self._get_value_from_ast(obj['slice'])).replace('(', '').replace(')', '').replace(' ', '').strip()}]"
        elif obj['_type'] == 'Slice':
            if obj.get('step'):
                return f"{self._get_value_from_ast(obj['lower']) if obj.get('lower', '') else ''}:" \
                       f"{self._get_value_from_ast(obj['upper']) if obj.get('upper', '') else ''}:" \
                       f"{self._get_value_from_ast(obj['step'])}"
            return f"{self._get_value_from_ast(obj['lower']) if obj.get('lower', '') else ''}:" \
                   f"{self._get_value_from_ast(obj['upper']) if obj.get('upper', '') else ''}:"
        elif obj['_type'] == 'BinOp':
            return self._parse_if_test(obj)
        elif obj['_type'] == 'UnaryOp':
            op = self.operators[obj['op']['_type']]
            new_op = f"{op} "
            return f"{op if op != 'not' else new_op}{self._get_value_from_ast(obj['operand'])}"
        elif obj['_type'] == 'Dict':
            keys = [x['value'] for x in obj['keys']]
            values = [y['value'] for y in obj['values']]
            return f"{dict(zip(keys, values))}"
        
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
            temp = ''
            closing = ''
            opening = ''
            alternate = True
            for ind, _ in enumerate(test['values']):
                if alternate:
                    left = self._parse_if_test(test['values'][ind])
                    alternate = not alternate
                    continue
                else:
                    right = self._parse_if_test(test['values'][ind])
                alternate = not alternate
                expression = f" {opening} {expression}{temp} ( {left} {op} {right} ) {closing} "
                temp = f'{op}'
                closing = ')'
                opening = '('
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