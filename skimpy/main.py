#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimpy import utils
import numpy as np

def skim(df,cat_variables):
    df_converted = utils.convert_to_cat(df,cat_variables)

    #############################################################
    # Dealing with Numeric Variables
    #############################################################
    df_num = df_converted.select_dtypes(include = [np.number])

    df_num_summary = df_num.apply(utils.apply_skimpy_col_numeric).T.reset_index()
    df_num_summary = df_num_summary.rename(columns = {"index": "var_name"})

    #############################################################
    # Dealing with Categorical Variables
    #############################################################

    df_cat = df_converted.select_dtypes(include = ['category'])

    df_cat_summary = df_cat.apply(utils.apply_skimpy_col_cat).T.reset_index()
    df_cat_summary = df_cat_summary.rename(columns = {"index": "var_name"})


    #############################################################
    # Output
    #############################################################
    print("-"*50)
    print("Numeric Variables")
    print("-"*50)
    print(df_num_summary.to_string())

    print("-"*50)
    print("Category Variables")
    print("-"*50)
    print(df_cat_summary.to_string())