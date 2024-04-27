import pandas as pd

def get_iris_data():
    
    filename = 'iris_data.csv'
    
    return pd.read_csv(filename, index_col=0)