import sys
import os
from xgboost import XGBClassifier
import logging
sys.path.append("./src")
from setup_logging import setup_logging
_DEFAULT_LOGGER_CONFIG_FILEPATH = "./loger_config/main_log.conf.yml"
setup_logging(_DEFAULT_LOGGER_CONFIG_FILEPATH)
logger = logging.getLogger("main_app")
logger_str = "predict_model"

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
#MODEL_RESULTS_FOLDER = '../../data/processed/xgb1'
MODEL_RESULTS_FOLDER = './data/processed/xgb1'
MODEL_SAVE_RESULTS_NAMES = [["y_train_pred", "y_train_proba"],
                            ["y_val_pred", "y_val_proba"],
                            ["y_test_pred", "y_test_proba"]]
MODEL_SAVE_RESULTS_PATH = [[os.path.join(MODEL_RESULTS_FOLDER, el[0]),
                            os.path.join(MODEL_RESULTS_FOLDER, el[1])]
                           for el in MODEL_SAVE_RESULTS_NAMES]


def main():
    logger.info("%s start predicting model", logger_str)
    loaded_datasets = []
    logger.debug("%s load datasets", logger_str)
    for ind, filepath in enumerate(INTERIM_DATASET_PATH):
        a = np.genfromtxt(filepath, delimiter=",")
        loaded_datasets.append(a)

    X_train, X_val, X_test, y_train, y_val, y_test = loaded_datasets

    logger.debug("%s load xgb model", logger_str)
    xgb = XGBClassifier()
    xgb.load_model(MODEL_SAVE_PATH)

    X_to_predict = [X_train, X_val, X_test]

    logger.debug("%s predict labels", logger_str)
    for ind, x in enumerate(X_to_predict):
        y_pred = xgb.predict(x)
        y_pred_proba = xgb.predict_proba(x)

        filepath1, filepath2 = MODEL_SAVE_RESULTS_PATH[ind]
        np.savetxt(filepath1, y_pred, delimiter=",")
        np.savetxt(filepath2, y_pred_proba, delimiter=",")

    logger.info("%s successfully predicting model", logger_str)

if __name__ == "__main__":
    main()