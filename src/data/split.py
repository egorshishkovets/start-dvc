"""
Команда запуска скрипта, параметры использованны по умолчанию,
запускается из корневой директории проекта
python src/data/split.py \
-x_train='data/interim/x_train.csv' \
-y_train='data/interim/y_train.csv' \
-path_out='data/processed'

Команда для запуска DVC
dvc run -n split \
-d src/data/split.py \
-d data/interim/x_train.csv \
-d data/interim/y_train.csv \
-o data/processed \
python src/data/split.py \
-x_train='data/interim/x_train.csv' \
-y_train='data/interim/y_train.csv' \
-path_out='data/processed'
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
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
    parser.add_argument('-path_out',
		         action="store",
		         dest="path_out")
    args = parser.parse_args()
    return args
args = get_args()
x_train = pd.read_csv(args.x_train)
y_train = pd.read_csv(args.y_train)
x_train,x_test,y_train,y_test=train_test_split(x_train, y_train, random_state=42)
os.makedirs(os.path.join(args.path_out.split('/')[0], args.path_out.split('/')[1]), exist_ok=True)
x_train.to_csv(f"{args.path_out}/x_train.csv", index=False)
y_train.to_csv(f"{args.path_out}/y_train.csv", index=False)
x_test.to_csv(f"{args.path_out}/x_test.csv", index=False)
y_test.to_csv(f"{args.path_out}/y_test.csv", index=False)
