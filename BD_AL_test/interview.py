import pandas as pd




def read_file(path: str) -> str:
    file = pd.read_csv(path)
    return file

# High performance python O'Reily

if __name__ == '__main__':
    file = read_file("/Users/mike/Downloads/conapesca.csv")
    new_file = file.to_json()
    #for index, key in enumerate(file.items(), start=1):
    #    print(f"{index}: {key[0]} {type(key[1][0])}")
    # nombre del indicador
    # unidad de medida
    # meta
    #print(f"All columns are: \n{file.columns} \n Unidad de medida {file['Unidad de Medida']} \n"
    #f"{file['Meta ']}")
    #print(file[['Meta ', 'Unidad de Medida']])
    sumatoria = lambda x: x + 2
    #print(file['Meta '])
    #print(list(map(sumatoria, file['Meta '])))
    
    def return_iter(file):
        for x in file['Unidad de Medida']:
            yield x 

    print(return_iter(file))

    iterc = return_iter(file)
    
    items = iter(iterc)
    no_more_values = True
    while no_more_values:
        try:
            next(items)
        except Exception as e:
            print("Ya no hay mas valores")
            no_more_values = False
    
            




