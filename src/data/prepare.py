"""
Команда запуска скрипта, параметры использованны по умолчанию,
запускается из корневой директории проекта
python src/data/prepare.py \
-path_train='data/raw/train.csv' \
-x_train='data/interim/x_train.csv' \
-y_train='data/interim/y_train.csv'

Команда для запуска DVC
dvc run -n prepare \
-d src/data/prepare.py \
-d data/raw \
-o data/interim/x_train.csv \
-o data/interim/y_train.csv \
python src/data/prepare.py \
-path_train='data/raw/train.csv' \
-x_train='data/interim/x_train.csv' \
-y_train='data/interim/y_train.csv'
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_origin',
                 action="store",
                 dest="path_train",
                 required=True)
    parser.add_argument('-path_prepared',
                 action="store",
                 dest="x_train",
                 required=True)
    args = parser.parse_args()
    return args


def prepare(path_file: str) -> (pd.DataFrame, pd.DataFrame):
   
    train = pd.read_csv(path_file)

    train_cat = list(train.select_dtypes(include='object'))
    train_num = list(train.select_dtypes(exclude='object'))

    train_missing_obj_col = []
    for col in train_cat:
        if train[col].isnull().any():
            train_missing_obj_col.append(col)
    # ['Cabin', 'Embarked']
    train_missing_num_col = []
    for col in train_num:
        if train[col].isnull().any():
            train_missing_num_col.append(col)
    # ['Age']
    temp = train[train_missing_num_col]
    train.drop(train_missing_num_col, inplace=True, axis=1)

    my_imputer = SimpleImputer()
    imputed_temp = pd.DataFrame(my_imputer.fit_transform(temp))
    imputed_temp.columns = temp.columns

    train = pd.concat([train, imputed_temp],axis=1)
    train['Embarked'].fillna(train['Embarked'].mode(),inplace=True)
    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']])
    train.drop(train_missing_obj_col, axis=1, inplace=True)
    train = pd.concat([train, dummy1],axis=1)
    train.drop(['Name', 'PassengerId', 'Sex', 'Ticket'], axis=1, inplace=True)
    train['Age'] = np.log(train['Age']+1)
    train['Fare'] = np.log(train['Fare']+1)
    
    y=train['Survived']
    train.drop(['Survived'],axis=1,inplace=True)

    return train, y

args = get_args()
x_train, y_train = prepare(args.path_train)

os.makedirs(os.path.join(args.x_train.split('/')[0], args.x_train.split('/')[1]), exist_ok=True)
x_train.to_csv(f"{args.x_train}/data.csv", index=False)
y_train.to_csv(f"{args.x_train}/target.csv", index=False)
