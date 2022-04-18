import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder


"""
Simple project to gain experience with data science and machine learning. The following sources were used:
    * Data cleaning guide: https://towardsdatascience.com/data-cleaning-in-python-the-ultimate-guide-2020-c63b88bf0a0d
    * Project suggestions: https://towardsdatascience.com/5-data-science-projects-in-healthcare-that-will-get-you-hired-81003cadf2f3
"""


def load_data(fname: str):
    """
    Reads in a csv file and return a DataFrame. A DataFrame df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)

def get_numeric_columns(df: pd.DataFrame):
    """
    Returns the names of the numeric columns within the DataFrame df.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    return numeric_cols

def get_non_numeric_columns(df: pd.DataFrame):
    """
    Returns the names of the non numeric columns within the DataFrame df.
    """
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    return non_numeric_cols

def create_ordinal_encoding_mappings(df: pd.DataFrame):
    """
    Iterates over categorical columns. Retrieves unique values in each column, and creates
    ordinal encoding mappings.
    """
    cols_list = []
    for column in df:
        unique_vals = df[column].unique()
        cols_list.append(unique_vals)
    return cols_list

def apply_ordinal_encoding_values_for_non_numeric_columns(df: pd.DataFrame, mappings_list: str, columns: list):
    """
    Applies ordinal encoding values to categorical dataframe.
    """
    encoder = OrdinalEncoder(categories=mappings_list)
    updated = encoder.fit_transform(df)
    df = pd.DataFrame(updated, columns=columns)
    return df, encoder

def remove_rows_with_null_bmi(df: pd.DataFrame):
    """
    Cleans the bmi data in DataFrame df by removing all rows where the bmi value is null.
    """
    df = df.drop(df[df.bmi.isnull()].index)
    return df

def evaluate_missing_values_in_df(df: pd.DataFrame):
    """
    Prints out the percent of values within a column that have null or N/A as the value.
    """
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing*100)))

def main():
    df = load_data('healthcare-dataset-stroke-data.csv')

    numeric_cols = get_numeric_columns(df)
    non_numeric_cols = get_non_numeric_columns(df)
    print(non_numeric_cols)

    print('Initial data evaluation...')
    evaluate_missing_values_in_df(df)
 
    df = remove_rows_with_null_bmi(df)

    print('Post data cleaning evaluation...')
    evaluate_missing_values_in_df(df)

    numerical_df = df[numeric_cols].copy()
    categorical_df = df[non_numeric_cols].copy()

    maps_list = create_ordinal_encoding_mappings(categorical_df)
    ec_applied_df, categories = apply_ordinal_encoding_values_for_non_numeric_columns(categorical_df, maps_list, non_numeric_cols)

    print("Ec applied datatypes...")
    print(ec_applied_df.dtypes)
    print(ec_applied_df)
    final_df = pd.merge(numerical_df, ec_applied_df, left_index=True, right_index=True)
    print(final_df)
    '''
    X_df = 
    y_df = 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    '''
    SVC_Gaussian = SVC(kernel='rbf')


if __name__ == '__main__':
    main()
