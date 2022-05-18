import sys
import os
from xgboost import XGBClassifier
import logging
sys.path.append("./src")
from setup_logging import setup_logging
_DEFAULT_LOGGER_CONFIG_FILEPATH = "./loger_config/main_log.conf.yml"
setup_logging(_DEFAULT_LOGGER_CONFIG_FILEPATH)
logger = logging.getLogger("main_app")
logger_str = "train_model"

import numpy as np


#INTERIM_DATA_FOLDER = '../../data/interim'
INTERIM_DATA_FOLDER = './data/interim'
NAMES_OF_INTERIM_DATASETS = ["X_train.csv", "X_val.csv", "X_test.csv",
                             "y_train.csv", "y_val.csv", "y_test.csv"]
INTERIM_DATASET_PATH = [os.path.join(INTERIM_DATA_FOLDER, el) for el in NAMES_OF_INTERIM_DATASETS]

#MODEL_SAVE_PATH = '../../models/xgb1.json'
MODEL_SAVE_PATH = './models/xgb1.json'
MODEL_OBJECTIVE = "binary:logistic"
MODEL_EVAL_METRIC = "auc"
MODEL_MAX_DEPTH = 1
MODEL_N_ESTIMATORS = 500
MODEL_LEARNING_RATE = 0.05
MODEL_COLSAMPLE_BYTREE = 0.35
MODEL_SUBSAMPLE = 0.5



def main():
    logger.info("%s start training model", logger_str)
    loaded_datasets = []
    logger.debug("%s load interim datasets", logger_str)
    for ind, filepath in enumerate(INTERIM_DATASET_PATH):
        a = np.genfromtxt(filepath, delimiter=",")
        loaded_datasets.append(a)

    X_train, X_val, X_test, y_train, y_val, y_test = loaded_datasets

    logger.debug("%s train xgb classifier", logger_str)
    xgb = XGBClassifier(objective=MODEL_OBJECTIVE, eval_metric=MODEL_EVAL_METRIC, max_depth=MODEL_MAX_DEPTH,
                        n_estimators=MODEL_N_ESTIMATORS, learning_rate=MODEL_LEARNING_RATE,
                        colsample_bytree=MODEL_COLSAMPLE_BYTREE, subsample=MODEL_SUBSAMPLE)
    xgb.fit(X_train, y_train)
    logger.debug("%s save model", logger_str)
    xgb.save_model(MODEL_SAVE_PATH)
    logger.info("%s successfully training model", logger_str)

if __name__ == "__main__":
    main()