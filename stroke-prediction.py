import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot

from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

def classification_feature_importance_CART(X: pd.DataFrame, y: pd.DataFrame, display_graph=False):
    """
    Reference: https://machinelearningmastery.com/calculate-feature-importance-with-python/
    """
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    feature_scores = []
    # summarize feature importance
    for i,v in enumerate(importance):
        feature_scores.append(v)
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    if display_graph:
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    return feature_scores

def classification_feature_importance_rand_forest(X: pd.DataFrame, y: pd.DataFrame, display_graph=False):
    # define the model
    model = RandomForestClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    feature_scores = []
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        feature_scores.append(v)
    # plot feature importance
    if display_graph:
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    return feature_scores

def compute_average_score(lhs: list, rhs: list):
    comb = []
    for i in range(0, len(lhs)):
        comb.append((lhs[i] + rhs[i]) / 2)
    return comb

def main():
    df = load_data('healthcare-dataset-stroke-data.csv')

    numeric_cols = get_numeric_columns(df)
    non_numeric_cols = get_non_numeric_columns(df)

    print('#' * 15, ' Initial data evaluation...', '#' * 15)
    evaluate_missing_values_in_df(df)

    print('#' * 15, ' Remove rows with null BMI...', '#' * 15)
    df = remove_rows_with_null_bmi(df)

    print('#' * 15, 'Post data cleaning evaluation...', '#' * 15)
    evaluate_missing_values_in_df(df)

    numerical_df = df[numeric_cols].copy()
    categorical_df = df[non_numeric_cols].copy()

    maps_list = create_ordinal_encoding_mappings(categorical_df)
    ec_applied_df, categories = apply_ordinal_encoding_values_for_non_numeric_columns(categorical_df, maps_list, non_numeric_cols)

    print('#' * 15, 'Ec applied datatypes...', '#' * 15)
    final_df = pd.merge(numerical_df, ec_applied_df, left_index=True, right_index=True)

    # Split into X, y sets
    y_df = final_df[['stroke']]
    X_df = final_df.drop(['stroke'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33)

    feature_index = {}
    idx = 0
    for feature in X_df.columns:
        feature_index[idx] = feature
        idx += 1

    scores_rand_forest = classification_feature_importance_rand_forest(X_df, y_df)
    scores_CART = classification_feature_importance_CART(X_df, y_df)

    # Using both of these measures, we can see that the most statistically significant features are features 1, 4, and 5.
    # Feature 0 appears to be significant, but it is just the ID number for the corresponding patient.
    average_scores = compute_average_score(scores_rand_forest, scores_CART)
    print('#' * 15, ' Enumerating average feature scores...', '#' * 15)
    for i,v in enumerate(average_scores):
        print('Feature: %0d, Score: %.5f' % (i,v))

    # Remaining steps:
    # * feature selection
    # * model selection and tuning
    # SVC_Gaussian = SVC(kernel='rbf')
    """
    Models to explore:
    * Logistic Regression
    * k-Nearest Neighbors
    * Decision Trees
    * Support Vector Machine
    * Naive Bayes
    """

if __name__ == '__main__':
    main()
