"""
Команда запуска скрипта, параметры использованны по умолчанию,
запускается из корневой директории проекта
python src/models/decision_tree_classifier.py \
-x_train='data/processed/x_train.csv' \
-y_train='data/processed/y_train.csv' \
-path_pkl='src/models/decision_tree_classifier.pkl'

Команда для запуска DVC
dvc run -n train_grid_search_decision_tree \
-p train_grid_search_decision_tree.criterion \
-p train_grid_search_decision_tree.max_depth \
-p train_grid_search_decision_tree.splitter \
-p train_grid_search_decision_tree.min_samples_leaf \
-p train_grid_search_decision_tree.min_samples_split \
-p train_grid_search_decision_tree.max_features \
-d src/models/decision_tree_classifier.py \
-d data/processed/x_train.csv \
-d data/processed/y_train.csv \
-o src/models/decision_tree_classifier.pkl \
python src/models/decision_tree_classifier.py \
-x_train='data/processed/x_train.csv' \
-y_train='data/processed/y_train.csv' \
-path_pkl='src/models/decision_tree_classifier.pkl'
"""
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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
params = yaml.safe_load(open('params.yaml'))['train_grid_search_decision_tree']
x_train = pd.read_csv(args.x_train)
y_train = pd.read_csv(args.y_train)
grid_param = {
    'criterion' : params['criterion'],
    'max_depth' : params['max_depth'],
    'splitter' : params['splitter'],
    'min_samples_leaf' : params['min_samples_leaf'],
    'min_samples_split' : params['min_samples_split'],
    'max_features' : params['max_features']
}
decision_tree = DecisionTreeClassifier()
decision = GridSearchCV(decision_tree, grid_param, cv = 5, n_jobs = -1, verbose = 1)
decision.fit(x_train, y_train.values.ravel())
print(decision.best_params_)
print(decision.best_score_)
decision_tree = DecisionTreeClassifier(criterion=decision.best_params_['criterion'],
                                       max_depth=decision.best_params_['max_depth'],
                                       splitter=decision.best_params_['splitter'],
                                       min_samples_leaf=decision.best_params_['min_samples_leaf'],
                                       min_samples_split=decision.best_params_['min_samples_split'],
                                       max_features=decision.best_params_['max_features'])
decision_tree.fit(x_train, y_train)
print(decision_tree.score(x_train,y_train.values.ravel()))
with open(args.path_pkl, 'wb') as fd:
    pickle.dump(decision_tree, fd)
