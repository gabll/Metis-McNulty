from sklearn.preprocessing import scale
import numpy as np
import pandas as pd


def convert_features(df, out_var, dummies=True, scaling=True, only_important_features=True, null_value=np.nan):
    """Convert features in a dataframe.
    df:      pandas dataframe (including outcome variable)
    out_var: outcome label (string)
    dummies: if True, convert all categorical features (string) to dummies
    scaling: if True, convert all continuous features to float type and then to z-score
    only_important_features: if True, it will keep the null_values as baseline. If false, it will drop the null_value rows
    null_value: null value used in the dataframe - string (ex: 'unknown')"""
    df = df.replace(null_value, np.nan)
    for i in df.columns:
    	if only_important_features == False:
    	    df = df.dropna(subset=[i])
        if df[i].dtypes == np.object and i != out_var and dummies:
            i_dummies = pd.get_dummies(df[i])
            i_dummies = i_dummies.add_prefix(i + '_')
            if df[i].nunique() == 2 and not np.any(df[i].isnull()):
                i_dummies = i_dummies.drop(i_dummies.columns[1], 1)
            df = pd.concat([df,i_dummies], axis=1)
            df = df.drop([i],1)   
        else:
            if scaling and i != out_var:
                df[i] = scale(df[i].astype('float'))
    return df.dropna()
