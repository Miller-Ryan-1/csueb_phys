import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Iris
def split_iris_data(df):
    train, test = train_test_split(df, train_size=.7, random_state=123, stratify=df.species)
    
    return train, test

def prep_iris(df):
    df = df.drop_duplicates()
    columns_to_drop = ['species_id','measurement_id']
    df = df.drop(columns = columns_to_drop)
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    df = pd.concat([df, dummy_df], axis=1)

    # train, validate, test = split_iris_data(df)
    
    # return train, validate, test
    return df