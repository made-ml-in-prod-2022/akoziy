import sys
from typing import Tuple
import os
import logging
sys.path.append("./src")
from setup_logging import setup_logging
_DEFAULT_LOGGER_CONFIG_FILEPATH = "./loger_config/main_log.conf.yml"
setup_logging(_DEFAULT_LOGGER_CONFIG_FILEPATH)
logger = logging.getLogger("main_app")
logger_str = "build_features"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#INTERIM_DATA_FOLDER = '../../data/interim'
INTERIM_DATA_FOLDER = './data/interim'
NAMES_OF_INTERIM_DATASETS = ["X_train.csv", "X_val.csv", "X_test.csv",
                             "y_train.csv", "y_val.csv", "y_test.csv"]
INTERIM_DATASET_PATH = [os.path.join(INTERIM_DATA_FOLDER, el) for el in NAMES_OF_INTERIM_DATASETS]
#RAW_DATASET_FOLDER = "../../data/raw"
RAW_DATASET_FOLDER = "./data/raw"
RAW_DATASET_PATH = os.path.join(RAW_DATASET_FOLDER, "heart_cleveland_upload.csv")


def make_train_valid_test(df: pd.DataFrame,
                            test_size: float,
                            val_size: float,
                            target_name: str,
                            random_state: int) -> Tuple[pd.DataFrame, ...]:
    data, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_name]
    )
    train, valid = train_test_split(
        data,
        test_size=val_size,
        random_state=random_state,
        stratify=data[target_name]
    )

    return train, valid, test


def main():
    logger.info("%s start build feature", logger_str)
    logger.debug("%s load dataset", logger_str)
    df = pd.read_csv(RAW_DATASET_PATH)


    logger.debug("%s prepare features", logger_str)
    # I had to create a new column since Seaborn gave me some trouble with proper legend labeling, it's not essential
    df["Heart condition"] = df["condition"].replace({1: "Present", 0: "Not present"})

    # Binning continous features together and therefore creating discrete categorical columns could
    # help the model to generalize the data and reduce overfitting
    df["thalach"] = pd.cut(df["thalach"], 8, labels=range(1, 9))
    df["trestbps"] = pd.cut(df["trestbps"], 5, labels=range(8, 13))
    df["age"] = pd.cut(df["age"], 12, labels=range(12, 24))
    df["chol"] = pd.cut(df["chol"], 10, labels=range(24, 34))
    df["oldpeak"] = pd.cut(df["oldpeak"], 5, labels=range(34, 39))

    a = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thalach", "trestbps", "age", "chol", "thal", "oldpeak"],
                       prefix=["cp", "restecg", "slope", "thalach", "trestbps", "age", "chol", "thal", "oldpeak"],
                       drop_first=True)

    a = a.drop("Heart condition", axis=1)

    X = a.drop(["condition", "restecg_1", "thalach_2", "thalach_8", "trestbps_12", "age_13", "age_22", "age_23",
                "chol_29", "chol_30", "chol_31", "chol_32", "chol_33", "oldpeak_37", "oldpeak_38"], axis=1)
    y = a.condition
    logger.debug("%s features are prepared", logger_str)

    logger.debug("%s split to train/val/test", logger_str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    datasets_for_dump = [X_train, X_val, X_test,
                         y_train, y_val, y_test]

    logger.debug("%s save dataset to path %s", logger_str, INTERIM_DATA_FOLDER)
    for ind, el in enumerate(datasets_for_dump):
        filepath = INTERIM_DATASET_PATH[ind]
        np.savetxt(filepath, el, delimiter=",")

    #print(X_train.shape)
    #print(y_train.shape)
    logger.info("%s successfully build features", logger_str)

if __name__ == '__main__':
    main()