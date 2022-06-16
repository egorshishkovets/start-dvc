"""
Команда запуска скрипта, параметры использованны по умолчанию,
запускается из корневой директории проекта
python src/models/logistic_regression.py \
-x_train='data/processed/x_train.csv' \
-y_train='data/processed/y_train.csv' \
-path_pkl='src/models/logistic_regression.pkl'

Команда для запуска DVC
dvc run -n train \
-p train_logistic.max_iter \
-p train_logistic.random_state \
-p train_logistic.n_jobs \
-d src/models/logistic_regression.py \
-d data/processed/x_train.csv \
-d data/processed/y_train.csv \
-o src/models/logistic_regression.pkl \
python src/models/logistic_regression.py \
-x_train='data/processed/x_train.csv' \
-y_train='data/processed/y_train.csv' \
-path_pkl='src/models/logistic_regression.pkl'
"""
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import yaml
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_train',
                 action="store",
                 dest="x_train",
                 required=True)
    parser.add_argument('-y_train',
                 action="store",
                 dest="y_train",
                 required=True)
    parser.add_argument('-path_pkl',
                 action="store",
                 dest="path_pkl")
    args = parser.parse_args()
    return args

args = get_args()
params = yaml.safe_load(open('params.yaml'))['train_logistic']
x_train = pd.read_csv(args.x_train)
y_train = pd.read_csv(args.y_train)
logistic=LogisticRegression(max_iter=params['max_iter'],
                            random_state=params['random_state'],
                            n_jobs=params['n_jobs'])
logistic.fit(x_train,y_train.values.ravel())
print(logistic.score(x_train,y_train))
with open(args.path_pkl, 'wb') as fd:
    pickle.dump(logistic, fd)
