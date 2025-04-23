from generator_gan import TableDataGenerator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import numpy as np
import random
from catboost import CatBoostClassifier
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

data = pd.read_csv("data/heart.xls")
data = data.drop_duplicates().reset_index(drop=True)

print(data)
print(data.dtypes)
print(data.isna().mean())

train, test = data[:100].to_numpy(), data[100:].to_numpy()

all_cols = list(data.columns)
cat_idx = [1, 2, 6, 13]
numeric_idx = [i for i in range(len(all_cols)) if i not in cat_idx]

scaler = MinMaxScaler()
train_scaled = train.copy()
train_scaled[:, numeric_idx] = scaler.fit_transform(train[:, numeric_idx])

test_scaled = test.copy()
test_scaled[:, numeric_idx] = scaler.fit_transform(test[:, numeric_idx])

# scaler = MinMaxScaler()
# train_scaled = np.nan_to_num(scaler.fit_transform(train), nan=-1.0)
# test_scaled = np.nan_to_num(scaler.transform(test), nan=-1.0)


generator = TableDataGenerator(train_scaled, cat_idx, numeric_idx, all_cols, optuna_timeout_seconds=40, n_optuna_trials_per_column=5000, train_epochs=10_000)

generated = generator.generate(3000)
#generated[generated < 0.0] = np.nan

#generated = scaler.inverse_transform(generated)
generated[:, numeric_idx] = scaler.inverse_transform(generated[:, numeric_idx])
generated = pd.DataFrame(generated, columns=all_cols)
generated[["sex", "cp", "restecg", "target"]] = generated[["sex", "cp", "restecg", "target"]].astype("int")

print("Generated: ")
print(generated)

normal_cat = CatBoostClassifier(verbose=0, random_state=42, cat_features=cat_idx[:-1])

train = pd.DataFrame(train, columns=data.columns)
train_scaled = pd.DataFrame(train_scaled, columns=data.columns)
test = pd.DataFrame(test, columns=data.columns)
test_scaled = pd.DataFrame(test_scaled, columns=data.columns)

train[["sex", "cp", "restecg", "target"]] = train[["sex", "cp", "restecg", "target"]].astype("int")
test[["sex", "cp", "restecg", "target"]] = test[["sex", "cp", "restecg", "target"]].astype("int")

normal_cat.fit(train.iloc[:, :-1], train_scaled.iloc[:, -1])
preds_test = normal_cat.predict(test.iloc[:, :-1])

scores = [[
    "Реальные данные",
    round(f1_score(test_scaled.iloc[:, -1], preds_test), 3),
    round(precision_score(test_scaled.iloc[:, -1], preds_test), 3),
    round(recall_score(test_scaled.iloc[:, -1], preds_test), 3),
    round(accuracy_score(test_scaled.iloc[:, -1], preds_test), 3),
]]

fake_cat = CatBoostClassifier(verbose=0, random_state=42, cat_features=cat_idx[:-1])
fake_cat.fit(generated.iloc[:, :-1], generated.iloc[:, -1])
preds_fake = fake_cat.predict(test.iloc[:, :-1])
scores.append([
    "Сгенерированные данные",
    round(f1_score(test_scaled.iloc[:, -1], preds_fake), 3),
    round(precision_score(test_scaled.iloc[:, -1], preds_fake), 3),
    round(recall_score(test_scaled.iloc[:, -1], preds_fake), 3),
    round(accuracy_score(test_scaled.iloc[:, -1], preds_fake), 3),
])

concat = pd.concat([train, generated])
print(f"{train.shape=}; {concat.shape=}")

combo_cat = CatBoostClassifier(verbose=0, random_state=42, cat_features=cat_idx[:-1])
combo_cat.fit(concat.iloc[:, :-1], concat.iloc[:, -1])
preds_combo = combo_cat.predict(test.iloc[:, :-1])
scores.append([
    "Реальные + сгенерированные данные",
    round(f1_score(test_scaled.iloc[:, -1], preds_combo), 3),
    round(precision_score(test_scaled.iloc[:, -1], preds_combo), 3),
    round(recall_score(test_scaled.iloc[:, -1], preds_combo), 3),
    round(accuracy_score(test_scaled.iloc[:, -1], preds_combo), 3),
])

scores = pd.DataFrame(scores, columns=["Тип данных", "F1", "Precision", "Recall", "Accuracy"])
print(scores)