import inspect

SUT_dir = './SUT/'+'test2.py'

data =''
with open('data.csv', 'r') as input_file:
    data = input_file.read()
    input_file.close()

print(data)
