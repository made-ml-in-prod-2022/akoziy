# ml_in_prod
MADE course ml in prod

Branch homework1 represents the first HW.
Guide for starting:

1. Prepare virtual environment

Create .venv in repository folder

```
python3.9 -m venv .venv
conda deactivate
source .venv/bin/activate
cd ml_project
pip install -r requirements.txt
```


2. Run the next scripts for performing the:

feature builder:
In repository currently exists dataset in data/raw/heart_cleveland_upload.csv
Feature builder will create datasets for X and y train/validation/test in data/interim folder
```
python src/features/build_features.py
```

train model:
Train model on train dataset and save model to /models folder
```
python src/models/train_model.py
```

predict model:
Load trained model and make a predictions on train/validation/test dataset. Predicted labels and probas saved to data/processed/<model_name>/
```
python src/models/predict_model.py
```

visualize results:
Load Predicted labels and probas for test dataset, and calculate some metrics. Also, it save figures to reports/<model_name>/ folder
```
python src/visualization/visualize.py
```


