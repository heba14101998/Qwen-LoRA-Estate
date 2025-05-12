# %% [code]
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def check_missing(df):
    """
    Check for missing and display them in a dataframe with:
    the feature name, the number of missing values, their percentage,
    their unique values, and the most common value.
    """
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing_percent = missing / len(df) * 100
        missing_df = pd.DataFrame({
            'feature': missing.index,
            'num_missing': missing.values,
            'percent_missing': missing_percent.values,
            'num_unique': df[missing.index].nunique().values,
            'most_common': df[missing.index].mode().iloc[0].values
        })
        return missing_df
    else:
        print("Dataset has no missing values")
        return 0

def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage reduced to {end_mem:.2f} MB ({(start_mem - end_mem)/start_mem:.1%} reduction)')
    return df

def check_outliers_zscore(df, threshold=3):
    """
    Check for outliers in numeric columns using Z-score method because the data is normally distributed.
    """

    numeric_cols = df.select_dtypes(include=np.number).columns
    outliers = {}
    
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers[col] = len(z_scores[z_scores > threshold])
    
    return pd.DataFrame({
        'feature': outliers.keys(),
        'num_outliers': outliers.values(),
        'percent_outliers (%)': (np.array(list(outliers.values()))) / len(df) * 100
    })


def timeit(func):
    """
    Decorator for timing the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n Data completed in {elapsed/60.0:.2f} minutes.")
        return result
    return wrapper

def logging_config(log_dir="/kaggle/working/logs"):
    """
    Configure logger with datetime-based filename
    """
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Also log to console if desired
        ]
    )
    return logging.getLogger(__name__)

def evaluate_model(true_labels, predicted_labels):
    """
    Evaluate the performance of a model using various regression metrics.
    """
    return {
        'MSE': mean_squared_error(true_labels, predicted_labels),
        'RMSE': mean_squared_error(true_labels, predicted_labels)**0.5,
        'MAE': mean_absolute_error(true_labels, predicted_labels),
        'R2': r2_score(true_labels,  predicted_labels)
    }