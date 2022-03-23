import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import get_db_url

def acquire_zillow():
    '''aquire zillow dataset from the database
    '''
    url = get_db_url('zillow')
    query = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            LEFT JOIN propertylandusetype USING(propertylandusetypeid)
            WHERE propertylandusedesc IN ("Single Family Residential",                       
                    "Inferred Single Family Residential")
            '''

    df = pd.read_sql(query, url)
    
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'square_ft_area',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built',
                          'taxamount': 'tax_amount'})
    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from an acquired dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([0.25, 0.75])
        
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr 
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def wrangle_zillow(df):
    ''' Prepare zillow dataset '''

    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_ft_area', 'tax_value', 'tax_amount'])
    
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    

    imputer = SimpleImputer(strategy='median')
    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test    