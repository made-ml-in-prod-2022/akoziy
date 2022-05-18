import sys
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import logging
sys.path.append("./src")
from setup_logging import setup_logging
_DEFAULT_LOGGER_CONFIG_FILEPATH = "./loger_config/main_log.conf.yml"
setup_logging(_DEFAULT_LOGGER_CONFIG_FILEPATH)
logger = logging.getLogger("main_app")
logger_str = "visualize"

import numpy as np


#INTERIM_DATA_FOLDER = '../../data/interim'
INTERIM_DATA_FOLDER = './data/interim'
NAMES_OF_INTERIM_DATASETS = ["X_train.csv", "X_val.csv", "X_test.csv",
                             "y_train.csv", "y_val.csv", "y_test.csv"]
INTERIM_DATASET_PATH = [os.path.join(INTERIM_DATA_FOLDER, el) for el in NAMES_OF_INTERIM_DATASETS]
MODEL_NAME = "xgb1"
#MODEL_SAVE_PATH = f'../../models/{MODEL_NAME}.json'
MODEL_SAVE_PATH = f'./models/{MODEL_NAME}.json'
MODEL_OBJECTIVE = "binary:logistic"
MODEL_EVAL_METRIC = "auc"
MODEL_MAX_DEPTH = 1
MODEL_N_ESTIMATORS = 500
MODEL_LEARNING_RATE = 0.05
MODEL_COLSAMPLE_BYTREE = 0.35
MODEL_SUBSAMPLE = 0.5
#MODEL_RESULTS_FOLDER = f'../../data/processed/{MODEL_NAME}'
MODEL_RESULTS_FOLDER = f'./data/processed/{MODEL_NAME}'
MODEL_SAVE_RESULTS_NAMES = [["y_train_pred", "y_train_proba"],
                            ["y_val_pred", "y_val_proba"],
                            ["y_test_pred", "y_test_proba"]]
MODEL_SAVE_RESULTS_PATH = [[os.path.join(MODEL_RESULTS_FOLDER, el[0]),
                            os.path.join(MODEL_RESULTS_FOLDER, el[1])]
                           for el in MODEL_SAVE_RESULTS_NAMES]
#FIGURES_SAVE_FOLDER = f"../../reports/{MODEL_NAME}"
FIGURES_SAVE_FOLDER = f"./reports/{MODEL_NAME}"
FIGURES_NAMES = ["fig1.png", "fig2.png"]
FIGURES_SAVE_PATH = [os.path.join(FIGURES_SAVE_FOLDER, el) for el in FIGURES_NAMES]

def main():
    logger.info("%s start visualize", logger_str)
    loaded_datasets = []
    logger.debug("%s load data", logger_str)
    for ind, filepath in enumerate(INTERIM_DATASET_PATH):
        a = np.genfromtxt(filepath, delimiter=",")
        loaded_datasets.append(a)

    X_train, X_val, X_test, y_train, y_val, y_test = loaded_datasets

    y_true = [y_train, y_val, y_test]
    y_pred_list = []
    y_pred_proba_list = []

    for ind, pathes in enumerate(MODEL_SAVE_RESULTS_PATH):
        filepath1, filepath2 = pathes
        pred = np.genfromtxt(filepath1, delimiter=",")
        pred_proba = np.genfromtxt(filepath2, delimiter=",")

        y_pred_list.append(pred)
        y_pred_proba_list.append(pred_proba)

    #print report
    logger.debug("%s make classification report", logger_str)
    print(classification_report(y_true[2], y_pred_list[2]))

    logger.debug("%s plor confusion matrix", logger_str)
    #plot cinfusion matrix
    sns.set_context("talk")
    plt.figure(figsize=(12, 9))
    sns.heatmap(confusion_matrix(y_true[2], y_pred_list[2]), annot=True, xticklabels=["Healthy", "Sick"],
                yticklabels=["Healthy", "Sick"], fmt="g", cmap="icefire_r")
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion matrix of xGBoosting")
    #plt.show()
    plt.savefig(FIGURES_SAVE_PATH[0], facecolor='y', bbox_inches="tight",
                pad_inches=0.3, transparent=True, dpi=100)

    logger.debug("%s plot roc-auc curve", logger_str)
    #xgb_prob = xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true[2], y_pred_proba_list[2][:, 1])
    sns.set_style("darkgrid")
    sns.set_context("poster")
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    sns.lineplot(fpr, tpr, alpha=0.6, ci=None)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(["Baseline", "xGBoosting"])
    #plt.show()
    plt.savefig(FIGURES_SAVE_PATH[1], facecolor='y', bbox_inches="tight",
                pad_inches=0.3, transparent=True, dpi=100)

    roc_auc = roc_auc_score(y_test, y_pred_proba_list[2][:, 1])
    logger.debug("%s print roc-auc %s", logger_str, roc_auc)
    print(roc_auc)

    logger.info("%s successfully visualize", logger_str)

if __name__ == "__main__":
    main()