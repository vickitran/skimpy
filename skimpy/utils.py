import re
import pandas as pd
import numpy as np

#############################################################
# Misc Utils
#############################################################

def build_sparkline(numbers):
    """
    https://github.com/holman/spark

    NOTE: Behavior is weird on Windows Machines!
    """
    try: bar = u'▁▂▃▄▅▆▇█'
    except: bar = '▁▂▃▄▅▆▇█'
    barcount = len(bar) - 1
    mn, mx = min(numbers), max(numbers)
    extent = mx - mn
    # if we pass in a uniform set of numbers
    # we might have division by zero
    if extent == 0:
        extent = 1
    sparkline = ''.join(bar[int( (n - mn) / extent * barcount)]
                        for n in numbers)

    return sparkline


def convert_to_cat(X,coltocat):
    """
    Convert all columns in coltocat into dtype category.

    See http://pandas.pydata.org/pandas-docs/stable/categorical.html

    TYPE:
        - transformation
    INPUT:
        - X : design matrix
        - coltocat - list of columns to convert
    OUTPUT:
        - X: modified design matrix
    """

    X_copy = X.drop(coltocat,1,errors = 'ignore')

    for col in coltocat:
        if col in X.columns:
            X_copy[col] = X[col].astype('category')

    return X_copy[X.columns]

def missing_data(s):
    """
    statistics regarding missing data in s

    INPUT:
        - s: data (expected input pandas series)
    OUTPUT:
        - {missing_vals: number_of_missing_entries, complete: number_of_complete_entries}
    """
    length_of_data = len(s)
    missing_vals = np.sum(s.isnull())
    return pd.Series([missing_vals, length_of_data-missing_vals], index = ["missing","complete"])

#############################################################
# Numeric Var Utils
#############################################################

def get_numeric_df(df):
    """
    tiny function to select numeric types from passed df

    """
    df_num = df.select_dtypes(include=[np.number])

    return df_num


def apply_skimpy_col_numeric(s):
    """
    handler function to build stats row

    INPUT:
        - s: data (expected input pandas series)
    OUTPUT:
        - row used in final output
          [missing_data, description_stats, histogram]

    """

    init = missing_data(s)
    init = init.append(pd.Series(s.describe()))
    init["hist"] =  build_sparkline(np.histogram(s)[0])
    return init


#############################################################
# Category Var Utils
#############################################################

def compute_value_counts(s):
    """
    Count how many instances of each category

    INPUT:
        - s: data (expected input pandas series)
    OUTPUT:
        - series of counts

    NOTE: Values more than 10, will be too much!

    """

    if s.nunique() < 10:
        return_val = [s.value_counts().to_dict()]
    else:
        return_val = "Unique Values {}".format(s.nunique())

    return pd.Series(return_val, index = ["n_unique"])

def apply_skimpy_col_cat(s):
    """
    handler function to build stats row

    INPUT:
        - s: data (expected input pandas series)
    OUTPUT:
        - row used in final output
        [missing_data, value_counts, histogram]

    NOTE: histogram is useful for seeing if a data set is unbalanced.
    """
    init = missing_data(s)
    init = init.append(compute_value_counts(s))
    init["hist"] =  build_sparkline(s.value_counts().values)
    return init