import os

def get_dim(SUT_dir = './SUT/'+'test.py'):
    ### Returns the number of arguments from the first function definition in a given python file
    data =''
    with open(SUT_dir, 'r') as input_file:
        data = input_file.read()
        input_file.close()

    program = data.split('def ')[1]
    ##TODO: Hay problema si después de la función hay código que está fuera?
    name = program[:program.find('(')]
    ##TODO: Add support for multiple functions in the same file
    args = program[program.find('(')+1:program.find(")")].split(',')
    #filtering *arguments
    arglist = [ arg for arg in args if '*' not in arg]
    
    return len(arglist), name

def getSUTS(SUT_dir):
    res = []
    # Iterate directory
    for file in os.listdir(SUT_dir):
        # check if current file is .py
        if file.endswith('.py'):
            res.append(file)
    return res
    


