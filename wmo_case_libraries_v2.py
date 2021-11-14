#-----------------------------------------------------------------------------------------------------------------------#
# JADS - WMO Case - Group 2 (year 2021)
#-----------------------------------------------------------------------------------------------------------------------#
# Set of procedures used in the jupyter notebook for the prediction model in regards to Social Support Act
#
#-----------------------------------------------------------------------------------------------------------------------#
# Author : Michiel Schakel
# Date   : October 2021
#-----------------------------------------------------------------------------------------------------------------------#

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

INTEGER_TYPES = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                 'uint32', 'uint64')
FLOAT_TYPES = ('float16', 'float32', 'float64')
NUMERIC_TYPES = INTEGER_TYPES + FLOAT_TYPES

A4_SIZE = (11.7, 8.3)
A3_SIZE = (16.5, 11.7)
JOINTPLOT_SIZE = 10


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


#-----------------------------------------------------------------------------------------------------------------------#
# strip_strings
#-----------------------------------------------------------------------------------------------------------------------#
def strip_strings(df):
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    return df


#-----------------------------------------------------------------------------------------------------------------------#
# remove_punctuations
#-----------------------------------------------------------------------------------------------------------------------#
def remove_punctuations(col):
    return ''.join([c for c in col if c not in set(punctuation) - set(['-'])])


#-----------------------------------------------------------------------------------------------------------------------#
# get_cbs_metadata
#-----------------------------------------------------------------------------------------------------------------------#
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


#-----------------------------------------------------------------------------------------------------------------------#
# get_cbs_data
#-----------------------------------------------------------------------------------------------------------------------#
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


#-----------------------------------------------------------------------------------------------------------------------#
# lower_memory_usage
#-----------------------------------------------------------------------------------------------------------------------#
def lower_memory_usage(df):
    print(f"Memory-usage before: {df.memory_usage().sum()}")

    df[df.select_dtypes(include="object").columns] = df.select_dtypes(
        include="object").astype("category")

    for old, new in [("integer", "unsigned"), ("float", "float")]:
        for col in df.select_dtypes(include=old).columns:
            df[col] = pd.to_numeric(df[col], downcast=new)

    print(f"Memory-usage after: {df.memory_usage().sum()}")

    return df


#-----------------------------------------------------------------------------------------------------------------------#
# get_corop_lists
#-----------------------------------------------------------------------------------------------------------------------#
def get_corop_lists(url):

    corop_matrix = pd.read_excel(url, index_col=None)

    corop_matrix = corop_matrix.loc[:,
                                    ['GM2017', 'GM2018', 'GM2019', 'GM2020']]
    corop_matrix.rename(columns={'GM2017': 'gm_code_origin'}, inplace=True)

    # Unpivot matrix to a list using het pd.melt function
    corop_list = pd.melt(corop_matrix,
                         id_vars=['gm_code_origin'],
                         var_name='year',
                         value_name='gm_code_new')

    # Remove 'GM' in front of the year and add GM in front of both the orginal municipality code and the new one
    corop_list['year'] = corop_list['year'].str[2:].astype(int)
    corop_list['gm_code_origin'] = 'GM' + corop_list['gm_code_origin'].astype(
        str).apply(lambda x: '{0:0>4}'.format(x))
    corop_list['gm_code_new'] = 'GM' + corop_list['gm_code_new'].astype(
        str).apply(lambda x: '{0:0>4}'.format(x))

    # Remove municipalities where no reorganization has taken place
    corop_list = corop_list[
        corop_list['gm_code_origin'] != corop_list['gm_code_new']]
    corop_list.sort_values(by=['year', 'gm_code_new'], inplace=True)
    corop_list.drop_duplicates(inplace=True)

    # Create list of unique municipality codes and names
    df_cbs_70072ned = get_cbs_data(
        datasets={0000: '70072ned'},
        filters=
        "startswith(RegioS,'GM') and Perioden ge '2017' and Perioden le '2020'",
        select={
            'KoppelvariabeleRegioCode_306': 'mun_code',
            'RegioS': 'mun_name'
        })

    df_cbs_70072ned = df_cbs_70072ned.drop_duplicates().dropna()

    return (corop_list, df_cbs_70072ned)


#-----------------------------------------------------------------------------------------------------------------------#
# remap_municipalities
#-----------------------------------------------------------------------------------------------------------------------#
def remap_municipalities(df_source, mun_code_col, mun_name_col, year_col,
                         df_corop_list, df_municipalities, do_remap):
    # Purpose: Remap municipalities that have been merged. Based on the COROP matrix given by the CBS, figures from
    #          municipalites that have been merged will be remapped to the new municipalitie.
    # Example of execution: remap_municipalities( cbs_data_merged, 'mun_code', 'mun_name', 'year', df_corop_list, df_municipalities, DO_REMAP )
    # Arguments:
    #   df_source         = The original dataframe
    #   mun_code_col      = The name of the column with the municipality codes
    #   mun_name_col      = The name of the column with the names of municipality
    #   year_col          = The name of the column with the years
    #   df_corop_list     = Dataframe with per year the original municipality code and the new one
    #   df_municipalities = Dataframe with code and name of all municipalities
    #   do_remap          = True or False whether the merged need to take place or note
    # Returns:
    #   df_total          = The remapped dataframe.

    df = df_source.copy()

    if not do_remap:
        print(f'Remapping skipped!')
    else:
        prev_year = 0000
        for index, row in df_corop_list.iterrows():
            # Print header for each new year
            if prev_year != row['year']:
                prev_year = row['year']
                print(
                    f"\nREORGANIZATIONS IN YEAR: {row['year']}\n-----------------------------"
                )
                print("Old".ljust(49) + "New")

            # Print code and name from both the original municipality and the destination.
            print(
                f"{row['gm_code_origin']} - {df_municipalities[df_municipalities['mun_code']==row['gm_code_origin']]['mun_name'].item().ljust(40) }",
                end='')
            print(
                f"{row['gm_code_new']} - {df_municipalities[df_municipalities['mun_code']==row['gm_code_new']]['mun_name'].item() }"
            )

            # Remap code of municipality
            df.loc[(df[mun_code_col] == row['gm_code_origin']) &
                   (df[year_col] <= row['year']),
                   mun_code_col] = row['gm_code_new']

            # Remap name of municipality
            val = df_municipalities[df_municipalities['mun_code'] ==
                                    row['gm_code_new']]['mun_name'].item()

            # Since the mun_name column is a categorie you need to add the new name as category first.
            # Otherwise you will get the error: "Cannot setitem on a Categorical with a new category, set the categories first"
            if val not in df[mun_name_col].cat.categories:
                df[mun_name_col].cat.add_categories(val, inplace=True)

            df.loc[(df[mun_code_col] == row['gm_code_new']),
                   mun_name_col] = val

            # Remove unused categories. If you do not remove them you'll issues during the groupby operations lateron.
            # This makes them appear again and will create extra null-records in your dataframe.
            df[mun_code_col] = df[mun_code_col].cat.remove_unused_categories()
            df[mun_name_col] = df[mun_name_col].cat.remove_unused_categories()

    return (df)


#-----------------------------------------------------------------------------------------------------------------------#
# check_columns
#-----------------------------------------------------------------------------------------------------------------------#
def check_columns(df, y_col, threshold_many_NA):
    # Purpose: Function to gather details of all predictors and whether a variable contains (a lot of) null values
    # Example of execution: check_columns( df, 'SalesPrice', 0.2)
    # Arguments:
    #   df                   = dataframe with all predictors (X's) and the response (Y)
    #   y_col                = name of the response variable in the dataframe
    #   threshold_many_NA    = threshold (%) to decide whether a column contains many null values
    # Returns:
    #   column_details       = dictionary with details (e.g. name, type, frequency) of all predictors
    #   columns_with_many_NA = list of column names of columns containing a log (> threshold) null values
    #   columns_with_NA      = list of column names of columns containing 1 or more null values

    y = df[y_col]

    # Create list of dictionaries with attributes of all columns in the dataframe
    column_details = [{
        'name': column,
        'anyNA': df[column].isna().any(),
        'type': df[column].dtypes,
        'frequency': df[column].value_counts().count(),
        'countNA': df[column].isna().sum(),
        'percNA': df[column].isna().sum() / len(df)
    } for column in df.drop([y_col], axis=1)]

    # Additional code-examples:
    # Print dictionary of specific column: print( [ d for d in column_details if d['name']=='Alley' ][0] )
    # Print attribute of dictionary of specific column: print( [ d for d in column_details if d['name']=='Alley' ][0]['percNA'] )

    # Create 4 lists of column names
    columns_with_many_NA = [
        d['name'] for d in column_details if d['percNA'] > threshold_many_NA
    ]

    columns_with_NA = [d['name'] for d in column_details if d['anyNA']]

    columns_numerical = [
        d['name'] for d in column_details if d['type'] in NUMERIC_TYPES
    ]

    columns_categorical = [
        d['name'] for d in column_details if d['type'].name == 'category'
    ]

    columns_numerical_with_NA = [
        d['name'] for d in column_details
        if d['anyNA'] and d['type'] in NUMERIC_TYPES
    ]

    columns_categorical_with_NA = [
        d['name'] for d in column_details
        if d['anyNA'] and d['type'].name == 'category'
    ]

    return (column_details, columns_with_NA, columns_with_many_NA,
            columns_numerical, columns_categorical, columns_numerical_with_NA,
            columns_categorical_with_NA)


#-----------------------------------------------------------------------------------------------------------------------#
# print_column_details
#-----------------------------------------------------------------------------------------------------------------------#
def print_column_details(df, column_details, y_col):
    # Purpose: to print all details of the featurs including an boxplot or jointplot diagram
    # Example of execution: print_column_details ( df, column_details, 'SalesPrice')
    # Arguments:
    #   df                   = dataframe with all predictors (X's) and the response (Y)
    #   column_details       = A list of dictionaries with all details of the features,
    #                          originates from the function check_colums and contains the details of all predictors
    #   y_col                = name of the response variable in the dataframe

    y = df[y_col]

    # Print for all columns either a jointplot (numeric) or a boxflot (categorical)
    for idx, row in enumerate(column_details):
        print(
            f"Column #    : {idx}\nColumn name : {color.RED + row['name'] + color.END}\nType        : {row['type']}\nNull values : {row['anyNA']} (#: {row['countNA']}, %: {row['percNA']:0.2%})\nFrequency   : {row['frequency']}\n"
        )

        if row['type'].name == 'category':
            print('Categorie count:')
            print(df[row['name']].value_counts().to_frame())

            plt.figure(figsize=(11.7, 2 + row['frequency'] * 0.70))

            sns.violinplot(data=df,
                           y=df[row['name']].sort_values(ascending=True),
                           x=y,
                           split=True,
                           inner="quart",
                           linewidth=1)
        elif row['type'].name in NUMERIC_TYPES:
            ax = sns.jointplot(x=df[row['name']],
                               y=y,
                               kind='reg',
                               height=JOINTPLOT_SIZE)

        plt.show()


#-----------------------------------------------------------------------------------------------------------------------#
# filter_dataframe
#-----------------------------------------------------------------------------------------------------------------------#
def filter_dataframe(df, filter_list):
    # Purpose: Function to gather details of all predictors and whether a variable contains (a lot of) null values
    # Arguments:
    #   df                   = dataframe with all predictors (X's) and the response (Y)
    #   filter_list          = list of dictionaries with the filterconditions
    #                          Example: [ {"column":"A", "operand":">", "value":0 }, {"column":"B", "operand":"in", "value":2 }, {"column":"C", "operand":"<=", "value":1 } ]
    #                          - Use one of the operands ['!=','==','<','>','<=','>=','in'] to filter based on value
    #                          - Use the operand drop to drop a column. Value have to be set to True
    #                          - Use the operand drop_col_list to drop a list of columns. Pass list of columns as dict-key "column". Set value to True
    #                          - Use the operand drop_na to drop rows with null values bases on a list of columns. The list of columns is used as subset in dropna()
    # Returns:
    #   df                   = filtered dataframe

    # Used as source: https://stackoverflow.com/questions/45925327/dynamically-filtering-a-pandas-dataframe

    df_return = df.copy()

    # Apply filter based on query
    query = ' & '.join([
        i['column'] + ' ' + i['operand'] + ' ' + str(i['value'])
        for i in filter_list
        if i['operand'] in ['!=', '==', '<', '>', '<=', '>=', 'in']
    ])
    if query:
        df_return = df_return.query(query)

    # Drop columns based on individual colums or a list of colums (e.g. columns with many NA)
    drop_columns = [
        i['column'] for i in filter_list
        if i['operand'] in ['drop'] and i['value']
    ]
    drop_col_list = [
        i['column'] for i in filter_list
        if i['operand'] in ['drop_col_list'] and i['value']
    ]
    drop_col_list = [
        waarde for sublijst in drop_col_list for waarde in sublijst
    ]  # Flatten list since it might contain lists in lists

    drop_columns += drop_col_list

    df_return = df_return.drop(columns=drop_columns)

    # Drop row with null values using a subset.
    drop_na = [
        i['column'] for i in filter_list
        if i['operand'] in ['drop_na'] and i['value']
    ]
    if len(drop_na) == 1:
        drop_na_subset = drop_na[0]
        df_return = df_return.dropna(subset=drop_na_subset)
    else:
        drop_na_subset = []

    # Print results
    print('FILTER DATAFRAME:')
    print('----------------')
    print(f'Filter applied            : {query}')
    print(f'Drop columns              : {drop_columns}')
    print(f'Drop null values (subset) : {drop_na_subset}')
    print()
    print('SHAPE (rows, columns): ')
    print(f'Shape before  : {df.shape}')
    print(f'Shape after   : {df_return.shape}')

    return (df_return)


#-----------------------------------------------------------------------------------------------------------------------#
# biplot
#-----------------------------------------------------------------------------------------------------------------------#
# taken from https://stackoverflow.com/a/46766116/3197404
def biplot(score, y, coeff, labels=None, plot_pc=(0, 1)):
    xs = score[:, plot_pc[0]]
    ys = score[:, plot_pc[1]]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.figure(figsize=(12, 12))
    plt.scatter(xs * scalex, ys * scaley, c=y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='y', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15,
                     coeff[i, 1] * 1.15,
                     "Var" + str(i + 1),
                     color='g',
                     ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15,
                     coeff[i, 1] * 1.15,
                     labels[i],
                     color='g',
                     ha='center',
                     va='center')
    plt.xlim(-0.75, 0.75)
    plt.ylim(-0.75, 0.75)
    plt.xlabel("Principle Component {}".format(plot_pc[0] + 1))
    plt.ylabel("Principle Component {}".format(plot_pc[1] + 1))
    plt.grid()


#-----------------------------------------------------------------------------------------------------------------------#
# print_heatmap_pairplot
#-----------------------------------------------------------------------------------------------------------------------#
def print_heatmap_pairplot(df, y_col, top_x, type='top', diagram='heatmap'):
    # Purpose: Print a heatmap with only the top X most correlated features.
    # Example of execution: print_heatmap( df_num, 10)
    # Arguments:
    #   df                   = dataframe with only numeric predictors (X's) and the response (Y)
    #   y_col                = name of response variable
    #   top_x                = the number of bars
    #   type                 = choose whether you want the top X or bottom X (top=default)
    #   diagram              = type of diagram 'heatmap' (default) or 'pairplot'

    sns.set(font_scale=1.2)
    if type == 'top':
        columns = df.corr().nlargest(top_x, y_col).index.values
    else:
        columns = df.corr().nsmallest(top_x, y_col).index.values
        columns = columns.tolist()
        columns.append(y_col) if y_col not in columns else columns

    if diagram == 'heatmap':
        # I used Pandas corr() in stead of np.corrcoef() since Pandas corr() is NaN friendly whereas NumPy not
        plt.figure(figsize=A3_SIZE)
        ax = plt.axes()
        sns.heatmap(df[columns].corr(),
                    cbar=True,
                    annot=True,
                    annot_kws={'size': 15},
                    yticklabels=columns,
                    xticklabels=columns,
                    ax=ax)
        ax.set_title(
            f'Heatmap {type.capitalize()} {top_x} correlated numeric features including response Y'
        )

    if diagram == 'pairplot':
        sns.set()
        sns.pairplot(df[columns], kind='reg', diag_kind='kde')
        #fig.subtitle(
        #    f'Pairplot {type.capitalize()} {top_x} correlated numeric features including response Y'
        #)

    plt.show()


#-----------------------------------------------------------------------------------------------------------------------#
# generate_scatter
#-----------------------------------------------------------------------------------------------------------------------#
def generate_scatter(df, hue_value):

    x_columns = list(df.select_dtypes(include=NUMERIC_TYPES).columns)
    x_select_box = alt.binding_select(options=x_columns, name='X-parameter: ')
    x_sel = alt.selection_single(fields=['x_column'],
                                 bind=x_select_box,
                                 init={'x_column': x_columns[0]})

    y_columns = list(df.select_dtypes(include=NUMERIC_TYPES).columns)
    y_select_box = alt.binding_select(options=y_columns, name='Y-parameter: ')
    y_sel = alt.selection_single(
        fields=['y_column'],
        bind=y_select_box,
        init={'y_column': y_columns[len(y_columns) - 2]})

    # selection of year for color/interactive legend selection
    sel_hue = alt.selection_multi(fields=[hue_value])
    # if you click on year in legend, rest will be gray; you can select multiple years by pressing shift
    color = alt.condition(
        sel_hue,
        alt.Color(hue_value + ':N',
                  legend=None,
                  scale=alt.Scale(scheme='category10')),
        alt.value('lightgray'))

    # make main chart
    chart = alt.Chart(df).transform_fold(
        x_columns,
        as_=['x_column',
             'x_parameter']).transform_filter(x_sel).transform_fold(
                 y_columns, as_=[
                     'y_column', 'y_parameter'
                 ]).transform_filter(y_sel).mark_point(opacity=0.4).encode(
                     y=alt.Y('y_parameter:Q',
                             axis=alt.Axis(title='Y-Parameter'),
                             scale=alt.Scale(zero=False)),
                     x=alt.X('x_parameter:Q',
                             axis=alt.Axis(title='X-Parameter'),
                             scale=alt.Scale(zero=False)),
                     color=color,
                     tooltip=[
                         alt.Tooltip('year:N', title='Jaar'),
                         alt.Tooltip('mun_name:N', title='Gemeente'),
                         alt.Tooltip('clients_per_1000_inhabitants:N',
                                     title='# Clients/1000 inh')
                     ]).add_selection(sel_hue).add_selection(
                         y_sel).add_selection(x_sel).properties(
                             width=400,
                             height=500,
                             title='Correlation between two variables')

    # make seperate 'plot' for legend
    leg = alt.Chart(df).mark_point().encode(
        y=alt.Y(hue_value + ':N', axis=alt.Axis(orient='right'), title=''),
        size=alt.value(200),
        color=color,
    ).add_selection(sel_hue)

    return (chart | leg).configure_title(fontSize=20,
                                         anchor='start',
                                         color='Black')


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_yearslider
#-----------------------------------------------------------------------------------------------------------------------#
# Color schemes: https://vega.github.io/vega/docs/schemes/
def generate_map_yearslider(df, gemeentes, legend_title, chart_title):

    # Pivot dataframe to create column per year.
    df_pivot = df.pivot(index='mun_name',
                        columns='year',
                        values='clients_per_1000_inhabitants').reset_index()
    df_pivot.columns = df_pivot.columns.map(str)

    min_val = max(15, min(df['clients_per_1000_inhabitants']))
    max_val = max(df['clients_per_1000_inhabitants'])

    columns = [
        str(year) for year in range(int(min(df['year'])),
                                    int(max(df['year'])) + 1)
    ]
    slider = alt.binding_range(min=int(min(df['year'])),
                               max=int(max(df['year'])),
                               step=1,
                               name='Selecteer een jaar:')
    select_year = alt.selection_single(name="Selector",
                                       fields=['year'],
                                       bind=slider,
                                       init={'year': int(max(df['year']))})

    cell_prefix = 'Y'

    chart = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='properties.statnaam',
            from_=alt.LookupData(df_pivot, 'mun_name', columns),
            default='200').transform_fold(columns, as_=[
                'year', cell_prefix
            ]).transform_calculate(year='parseInt(datum.year)').encode(
                tooltip=[
                    alt.Tooltip('properties.statnaam:N', title="Gemeente"),
                    alt.Tooltip(cell_prefix + ':Q', title=legend_title),
                    alt.Tooltip('year:Q', title='Jaar')
                ],
                color=alt.condition(
                    'datum.' + cell_prefix + ' > 0',
                    alt.Color(cell_prefix + ':Q',
                              scale=alt.Scale(scheme='yelloworangered',
                                              type='symlog',
                                              domain=[min_val, max_val]),
                              sort='ascending',
                              legend=alt.Legend(orient='top',
                                                title=legend_title,
                                                gradientLength=330,
                                                tickCount=4,
                                                titleLimit=200)),
                    alt.value('#dbe9f6'))).add_selection(
                        select_year).properties(
                            width=400, height=500,
                            title=chart_title).transform_filter(
                                select_year).configure_view(
                                    strokeWidth=0).configure_title(
                                        fontSize=20,
                                        anchor='start',
                                        color='Black')

    return (chart)


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_value
#-----------------------------------------------------------------------------------------------------------------------#
def generate_map_value(df, gemeentes, value_col, title):

    max_val = max(df.loc[~df[value_col].isna(), value_col])
    min_val = min(df.loc[~df[value_col].isna(), value_col])

    chart = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='properties.statnaam',
            from_=alt.LookupData(df, 'mun_name', [value_col]),
            default='200').encode(
                tooltip=[
                    alt.Tooltip('properties.statnaam:N', title="Gemeente"),
                    alt.Tooltip(value_col + ':Q', title=title)
                ],
                color=alt.Color(value_col + ':Q',
                                scale=alt.Scale(scheme='yelloworangered',
                                                type='linear',
                                                domain=[min_val, max_val]),
                                sort='ascending',
                                legend=alt.Legend(
                                    orient='top',
                                    title=title,
                                    gradientLength=330,
                                    tickCount=4,
                                    titleLimit=200))).properties(
                                        width=400, height=500,
                                        title=title).configure_title(
                                            fontSize=20,
                                            anchor='start',
                                            color='Black')

    return (chart)


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_groupedvalue
#-----------------------------------------------------------------------------------------------------------------------#
def generate_map_groupedvalue(df, gemeentes, value_col, title, cut_bins,
                              cut_labels):

    df['grp_value'] = pd.cut(df[value_col], bins=cut_bins, labels=cut_labels)
    df['grp_value'].cat.add_categories('0. missing', inplace=True)
    df['grp_value'].fillna('0. missing', inplace=True)
    cut_labels = ['0. missing'] + cut_labels

    color = alt.Color('grp_value:N',
                      scale=alt.Scale(scheme='turbo'),
                      sort=cut_labels,
                      legend=alt.Legend(orient='top',
                                        title=title,
                                        gradientLength=330,
                                        tickCount=4,
                                        titleLimit=200))

    chart = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='properties.statnaam',
            from_=alt.LookupData(df, 'mun_name', ['grp_value']),
            default='0. missing').encode(tooltip=[
                alt.Tooltip('properties.statnaam:N', title="Gemeente"),
                alt.Tooltip('grp_value:N', title=title)
            ],
                                         color=color).properties(
                                             width=400,
                                             height=500,
                                             title=title).configure_title(
                                                 fontSize=20,
                                                 anchor='start',
                                                 color='Black')

    return (chart)


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_with_barchart
#-----------------------------------------------------------------------------------------------------------------------#
def generate_map_with_barchart(df, gemeentes, legend_dict, value_col,
                               barchart_cols, title, cut_bins, cut_labels):

    df['grp_value'] = pd.cut(df[value_col], bins=cut_bins, labels=cut_labels)
    df['grp_value'].cat.add_categories('0. missing', inplace=True)
    df['grp_value'].fillna('0. missing', inplace=True)
    cut_labels = ['0. missing'] + cut_labels

    legend_dict['grp_value:N'] = 'Value (grp)'

    # selection based on click on the map
    single = alt.selection_single(fields=['mun_name'])  #, bind='legend')

    color = alt.Color('grp_value:N',
                      scale=alt.Scale(scheme='turbo'),
                      sort=cut_labels,
                      legend=alt.Legend(orient='top',
                                        title=title,
                                        gradientLength=330,
                                        tickCount=4,
                                        titleLimit=200))

    # make the map and color based on std groups
    ch1 = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='id',
            from_=alt.LookupData(
                df, 'mun_code',
                [key[0:-2] for (key, val) in legend_dict.items()]),
            default='0. missing').encode(
                tooltip=[
                    alt.Tooltip(shorthand=key, title=val)
                    for (key, val) in legend_dict.items()
                ],
                color=color,
                opacity=alt.condition(
                    single, alt.value(1),
                    alt.value(0.3))).add_selection(single).properties(
                        width=400, height=500, title=title)

    # extract the data per year again and change by selecting the gemeente
    bar = alt.Chart(df).transform_fold(barchart_cols, ).mark_bar().encode(
        y='key:N',
        x='value:Q',
    ).transform_filter(single)

    # add textlabels to barchart
    text = bar.mark_text(align='left', baseline='middle', dx=3).encode(
        text=alt.Text('value:Q', format=',.2f')).transform_filter(single)

    # show them next to each other
    return (ch1 | bar + text).configure_title(fontSize=20,
                                              anchor='start',
                                              color='Black')


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_valueslider
#-----------------------------------------------------------------------------------------------------------------------#
def generate_map_valueslider(df,
                             gemeentes,
                             value_col,
                             title,
                             legend_title,
                             step_size=0.1):

    # Make copy of valuecolumn since it was not possible to work with a dynamically named value column in the slider condition
    df['tmp_value'] = df[value_col]
    max_val = np.round(max(df.loc[~df['tmp_value'].isna(), 'tmp_value']),
                       5) + 0.1
    min_val = np.round(min(df.loc[~df['tmp_value'].isna(), 'tmp_value']), 5)

    slider = alt.binding_range(min=min_val,
                               max=max_val,
                               step=step_size,
                               name='cutoff:')
    selector = alt.selection_single(name="SelectorName",
                                    fields=['cutoff'],
                                    bind=slider,
                                    init={'cutoff': min_val})

    chart = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='properties.statnaam',
            from_=alt.LookupData(df, 'mun_name', ['tmp_value']),
            default='90').encode(
                tooltip=[
                    alt.Tooltip('properties.statnaam:N', title="Gemeente"),
                    alt.Tooltip('tmp_value:Q', title=legend_title)
                ],
                color=alt.condition(
                    alt.datum.tmp_value < selector.cutoff,
                    alt.Color('tmp_value:Q',
                              scale=alt.Scale(scheme='yellowgreen',
                                              type='linear',
                                              domain=[min_val, max_val]),
                              sort='descending',
                              legend=alt.Legend(orient='top',
                                                title=legend_title,
                                                gradientLength=330,
                                                tickCount=4,
                                                titleLimit=200)),
                    alt.value('red'))).properties(
                        width=400, height=500, title=title).configure_title(
                            fontSize=20, anchor='start',
                            color='Black').add_selection(selector)

    return (chart)


#-----------------------------------------------------------------------------------------------------------------------#
# generate_map_valueslider_redgreen
#-----------------------------------------------------------------------------------------------------------------------#
def generate_map_valueslider_redgreen(df, gemeentes, value_col, title,
                                      legend_title):

    df['tmp_value'] = df[value_col]
    max_val = np.round(max(df.loc[~df['tmp_value'].isna(), 'tmp_value']),
                       5) + 0.1
    min_val = np.round(min(df.loc[~df['tmp_value'].isna(), 'tmp_value']), 5)

    slider = alt.binding_range(min=min_val,
                               max=max_val,
                               step=0.1,
                               name='cutoff:')
    selector = alt.selection_single(name="SelectorName",
                                    fields=['cutoff'],
                                    bind=slider,
                                    init={'cutoff': min_val})

    chart = alt.Chart(gemeentes).mark_geoshape(
        stroke='black', strokeWidth=0.05).transform_lookup(
            lookup='properties.statnaam',
            from_=alt.LookupData(df, 'mun_name', ['tmp_value']),
            default='null').encode(
                tooltip=[
                    alt.Tooltip('properties.statnaam:N', title="Gemeente"),
                    alt.Tooltip('tmp_value:Q', title=legend_title)
                ],
                color=alt.condition(
                    alt.datum.tmp_value < selector.cutoff, alt.value('green'),
                    alt.value('red'))).properties(
                        width=400, height=500, title=title).configure_title(
                            fontSize=20, anchor='start',
                            color='Black').add_selection(selector)

    return (chart)


#-----------------------------------------------------------------------------------------------------------------------#
# generate_barchart_per_municipality
#-----------------------------------------------------------------------------------------------------------------------#
def generate_barchart_per_municipality(df, value_col, title):

    df_pivot_mun = df.pivot(index='year', columns='mun_name', values=value_col)

    # Convert columnnames to strings and reset index afterwards
    df_pivot_mun.columns = df_pivot_mun.columns.astype(str)
    df_pivot_mun = df_pivot_mun.reset_index()

    # Remove punctuations from column names
    res = {
        c: remove_punctuations(c)
        for c in df_pivot_mun.columns if set(punctuation) & set(c) - set(['-'])
    }
    df_pivot_mun.rename(columns=res, inplace=True)

    gemeenten = sorted([
        c for c in list(df_pivot_mun.columns)
        if len(c) < 30 and c not in ['year']
    ])
    select_box = alt.binding_select(options=gemeenten, name='Gemeente: ')
    sel = alt.selection_single(fields=['Gemeente'],
                               bind=select_box,
                               init={'Gemeente': 'Gemert-Bakel'})

    chart = alt.Chart(df_pivot_mun).transform_fold(
        gemeenten,
        as_=['Gemeente', 'value']).transform_filter(sel).mark_bar().encode(
            x=alt.X('year:O', title='Jaar'),
            y=alt.X('value:Q', title='Aantal'),
            tooltip=[
                alt.Tooltip('Jaar:O', title="Jaar"),
                alt.Tooltip('value:Q', title='Waarde')
            ],
        ).add_selection(sel).properties(
            width=500, height=325,
            title=title).configure_view(strokeWidth=0).configure_title(
                fontSize=20, anchor='start', color='Black')

    return chart


#-----------------------------------------------------------------------------------------------------------------------#
# shift_num_clients_1year
#-----------------------------------------------------------------------------------------------------------------------#
def shift_num_clients_1year(df):
    # Shift the client_per_1000_inhabitants with one year. Since this has to be done per municipality a trick is performed to create some sort of group per municipality
    # Based on example: https://dfrieds.com/data-analysis/shift-method-python-pandas.html
    #df = cbs_data_merged.copy()

    # Create a column where we shift the municipality code with -1 periods
    df.sort_values(by=['mun_code', 'year'], inplace=True)
    df['prev_mun_code'] = df['mun_code'].shift(periods=-1)

    # Create a column clients_per_1000_inhabitants_ny with the value of clients_per_1000_inhabitants from the next observation/record (shift period -1)
    df['clients_per_1000_inhabitants_ny'] = df[
        'clients_per_1000_inhabitants'].shift(periods=-1)

    # Filter only the records where prev_mun_code and mun_code are the same. This will result in loosing the observations of the last year (2020)
    df = df.where(df['prev_mun_code'] == df['mun_code'])

    #df_test.loc[df_test['mun_name']=='Tilburg',['year','mun_code','prev_mun_code','mun_name','clients_per_1000_inhabitants','clients_per_1000_inhabitants_ny']].head()
    df.drop(columns='prev_mun_code', inplace=True)

    #df[columns_categorical_with_NA]
    df['clients_per_1000_inhabitants_ny'].fillna(
        value=df['clients_per_1000_inhabitants_ny'].mean(), inplace=True)

    return (df)


#-----------------------------------------------------------------------------------------------------------------------#
# split_column_over_years
#-----------------------------------------------------------------------------------------------------------------------#
def split_columns_over_years(df,
                             split_columns,
                             index_column,
                             split_by,
                             base_year,
                             y,
                             intpl=False):

    min_year = df[split_by].min()
    max_year = df[split_by].max()
    max_year_min1 = df[split_by].max() - 1

    # Create dataframe for all other columns than the y column since we need to exclude the last year (max) and pivot
    temp1 = df.loc[df[split_by] != max_year, [index_column, split_by] +
                   [x for x in split_columns if x != y]]
    tmp_pivot1 = temp1.pivot(index=index_column,
                             columns=split_by,
                             values=[x for x in split_columns
                                     if x != y]).reset_index()

    # Create dataframe for the y columns since we need all years and pivot
    temp2 = df.loc[:, [index_column, split_by] +
                   [x for x in split_columns if x == y]]
    tmp_pivot2 = temp2.pivot(index=index_column,
                             columns=split_by,
                             values=[x for x in split_columns
                                     if x == y]).reset_index()

    # Rename the columns
    tmp_pivot1.columns = [
        ''.join(map(str, col)).strip() for col in tmp_pivot1.columns.values
    ]
    tmp_pivot2.columns = [
        ''.join(map(str, col)).strip() for col in tmp_pivot2.columns.values
    ]

    df_return = df.loc[df[split_by] == base_year,
                       [x for x in df.columns if not x in split_columns]]

    df_return = df_return.merge(tmp_pivot1, on=index_column)
    df_return = df_return.merge(tmp_pivot2, on=index_column)

    df_return['num_NaN_WMO'] = df_return.loc[:, y + str(min_year):y +
                                             str(max_year_min1)].isna().sum(
                                                 axis=1)

    # add interpolation if intpl == True before you calculate std etc.
    if intpl:
        for col in split_columns:
            df_return.loc[:, col + str(min_year):col +
                          str(max_year_min1)] = df_return.loc[:, col + str(
                              min_year):col + str(max_year_min1)].interpolate(
                                  axis=1,
                                  method='linear',
                                  limit_direction='both')

    # Add slope and intercept (by Marcel)
    A = np.vstack([np.arange(0, 3), np.ones(3)]).T
    for col in split_columns:
        # Calculate slope
        df_return[col + '_slope'] = [
            np.linalg.lstsq(A,
                            df_return.loc[:, col + str(min_year):col +
                                          str(max_year_min1)].values[i],
                            rcond=None)[0][0] for i in range(len(df_return))
        ]

        # Calculate intercept
        df_return[col + '_intercept'] = [
            np.linalg.lstsq(A,
                            df_return.loc[:, col + str(min_year):col +
                                          str(max_year_min1)].values[i],
                            rcond=None)[0][1] for i in range(len(df_return))
        ]

        # Calculate standard deviation
        df_return[col + '_std'] = df_return.loc[:, col + str(min_year):col +
                                                str(max_year_min1)].std(axis=1)

        # Calculate standard deviation (normalized)
        df_return[col + '_std_norm'] = df_return.loc[:, col + str(
            min_year):col + str(max_year_min1)].std(
                axis=1) / df_return.loc[:, col + str(min_year):col +
                                        str(max_year_min1)].mean(axis=1)

    return (df_return)


#-----------------------------------------------------------------------------------------------------------------------#
# prepare_dataset
#-----------------------------------------------------------------------------------------------------------------------#
def prepare_dataset(df, y, missing, test_size, random_state):

    tmp_df = lower_memory_usage(df)

    df_train, df_test = train_test_split(tmp_df,
                                         test_size=test_size,
                                         random_state=random_state)

    X_train = df_train.drop(columns=[y])
    y_train = df_train[y]
    #y_train = np.log(df_train[Y_COL])

    # Same for test set
    X_test = df_test.drop(columns=[y])
    y_test = df_test[y]
    #y_test = np.log(df_test[Y_COL])

    # Using the log() gave the error "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')."
    # Probably due to the fact that there are zero's in the dataset.
    # Have to check: https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32

    cat_cols = X_train.select_dtypes(include='category').columns
    num_cols = X_train.select_dtypes(include='number').columns

    # Fill missing values for categorical features with the value given by the constante parameter missing

    # First you have to add the constante value to the available categories in each categorical column
    # Otherwise you'll get an error "fill value must be in categories" in the next command.

    for col in cat_cols:
        if missing not in tmp_df[col].cat.categories:
            tmp_df[col].cat.add_categories(missing, inplace=True)

    categories = [tmp_df[column].cat.categories for column in cat_cols]

    print(f'\nShape dataframe : {tmp_df.shape}')
    print(f'Shape X_train   : {X_train.shape}')
    print(f'Shape X_test    : {X_test.shape}')
    print(f'Shape y_train   : {y_train.shape}')
    print(f'Shape y_test    : {y_test.shape}')

    return (tmp_df, X_train, y_train, X_test, y_test, cat_cols, num_cols,
            categories)
