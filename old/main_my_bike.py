from generator import TableDataGenerator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
import random
from catboost import CatBoostRegressor
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

data = pd.read_csv("data/bikeshare.csv")
data = data.drop_duplicates().reset_index(drop=True)
data = data.drop(columns=["instant", "dteday", "casual", "registered"])

print(data)
print(data.dtypes)
print(data.isna().mean())

train, test = data[:100].to_numpy(), data[100:].to_numpy()

all_cols = list(data.columns)
cat_idx = [0, 1, 2, 3, 4, 5, 6]
numeric_idx = [i for i in range(len(all_cols)) if i not in cat_idx]

scaler = MinMaxScaler()
train_scaled = train.copy()
train_scaled[:, numeric_idx] = scaler.fit_transform(train[:, numeric_idx])

test_scaled = test.copy()
test_scaled[:, numeric_idx] = scaler.fit_transform(test[:, numeric_idx])

# scaler = MinMaxScaler()
# train_scaled = np.nan_to_num(scaler.fit_transform(train), nan=-1.0)
# test_scaled = np.nan_to_num(scaler.transform(test), nan=-1.0)


generator = TableDataGenerator(train_scaled, cat_idx, numeric_idx, all_cols, optuna_timeout_seconds=40, n_optuna_trials_per_column=5000, train_epochs=25_000)

generated = generator.generate(3_000_000)
#generated[generated < 0.0] = np.nan

#generated = scaler.inverse_transform(generated)
generated[:, numeric_idx] = scaler.inverse_transform(generated[:, numeric_idx])
generated = pd.DataFrame(generated, columns=all_cols)
generated[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]] = generated[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]].astype("int")

print("Generated: ")
print(generated)

normal_cat = CatBoostRegressor(verbose=0, random_state=42, cat_features=cat_idx[:-1])

train = pd.DataFrame(train, columns=data.columns)
train_scaled = pd.DataFrame(train_scaled, columns=data.columns)
test = pd.DataFrame(test, columns=data.columns)
test_scaled = pd.DataFrame(test_scaled, columns=data.columns)

train[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]] = train[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]].astype("int")
test[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]] = test[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]].astype("int")

normal_cat.fit(train.iloc[:, :-1], train_scaled.iloc[:, -1])
preds_test = normal_cat.predict(test.iloc[:, :-1])

scores = [[
    "Реальные данные",
    round(mean_absolute_error(test_scaled.iloc[:, -1], preds_test), 3),
    round(mean_absolute_percentage_error(test_scaled.iloc[:, -1], preds_test) * 100.0, 3),
    round(r2_score(test_scaled.iloc[:, -1], preds_test), 3),
]]

fake_cat = CatBoostRegressor(verbose=0, random_state=42, cat_features=cat_idx[:-1])
fake_cat.fit(generated.iloc[:, :-1], generated.iloc[:, -1])
preds_fake = fake_cat.predict(test.iloc[:, :-1])
scores.append([
    "Сгенерированные данные",
    round(mean_absolute_error(test_scaled.iloc[:, -1], preds_fake), 3),
    round(mean_absolute_percentage_error(test_scaled.iloc[:, -1], preds_fake) * 100.0, 3),
    round(r2_score(test_scaled.iloc[:, -1], preds_fake), 3),
])

concat = pd.concat([train, generated])
print(f"{train.shape=}; {concat.shape=}")

combo_cat = CatBoostRegressor(verbose=0, random_state=42, cat_features=cat_idx[:-1])
combo_cat.fit(concat.iloc[:, :-1], concat.iloc[:, -1])
preds_combo = combo_cat.predict(test.iloc[:, :-1])
scores.append([
    "Реальные + сгенерированные данные",
    round(mean_absolute_error(test_scaled.iloc[:, -1], preds_combo), 3),
    round(mean_absolute_percentage_error(test_scaled.iloc[:, -1], preds_combo) * 100.0, 3),
    round(r2_score(test_scaled.iloc[:, -1], preds_combo), 3),
])

scores = pd.DataFrame(scores, columns=["Тип данных", "MAE", "MAPE, %", "R2"])
print(scores)