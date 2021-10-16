import requests
import cbsodata
import pandas as pd  # Library to work with dataframes
from string import punctuation  # String manipulation
import altair as alt  # Create interactive charts and maps

import seaborn as sns  # especially for pairplots
import numpy as np  # fill diagonal...corr plot
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_predict, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def strip_strings(df):
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    return df


def get_cbs_metadata(dataset):
    # Purpose: Getting and printing CBS metadata for a specific datasets (e.g. 84751NED)
    #          Info and available datasets: https://opendata.cbs.nl/statline/portal.html?_la=nl
    # Example of execution: check_columns( { 2019: '84753NED', 2020: '84908NED' } )
    # Arguments:
    #   dataset              = stringvalue that refers to the CBS dataset (e.g. '84908NED')
    # Returns:
    #   l_metadata           = dataframe with metadata of the given dataset. Some details are printed as well.

    # Downloading table list
    # toc = pd.DataFrame(cbsodata.get_table_list())

    # Downloading metadata
    l_metadata = pd.DataFrame(cbsodata.get_meta(dataset, "DataProperties"))
    print(l_metadata[["Key", "Title", "Type"]])

    return l_metadata


def get_cbs_data(datasets, filters, select):
    # Purpose: Getting CBS data for several similar datasets (e.g. same dataset over several years)
    # Example of execution: check_columns( { 2019: '84753NED', 2020: '84908NED' } )
    # Arguments:
    #   datasets  = dictionary with keys = year and values = dataset
    #   filters   = filter provided to odata-request
    #   select    = dictionary with keys = CBS-columns to select, values = new name of columns
    # Returns:
    #   df_total  = dataframe with CBS over multiple years.

    list_of_df = []

    select_list = [column_o for (column_o, column_n) in select.items()]
    rename_list = ["year"] + \
        [column_n for (column_o, column_n) in select.items()]

    for (year, dataset) in datasets.items():
        print(f"Retrieving CBS-data - year: {year} (dataset: {dataset})...",
              end="")

        df = pd.DataFrame(
            cbsodata.get_data(dataset,
                              filters=filters.format(year),
                              select=select_list))

        # Add year to the dataframe as first column
        df.insert(0, "year", year)

        print(f"  ready - size: {df.shape}")

        # Add dataframe to list and release memory
        list_of_df.append(df)
        del df

    print("Concatenate...")
    df_total = pd.concat(list_of_df, ignore_index=True)

    print("Rename columns...")
    df_total.columns = rename_list

    print("Remove all leading and trailing spaces...")
    df_total = strip_strings(df_total)

    print(f"Ready, size: {df_total.shape}")

    return df_total


def lower_memory_usage(df):
    print(f"Memory-usage before: {df.memory_usage().sum()}")

    df[df.select_dtypes(include="object").columns] = df.select_dtypes(
        include="object").astype("category")

    for old, new in [("integer", "unsigned"), ("float", "float")]:
        for col in df.select_dtypes(include=old).columns:
            df[col] = pd.to_numeric(df[col], downcast=new)

    print(f"Memory-usage after: {df.memory_usage().sum()}")

    return df
