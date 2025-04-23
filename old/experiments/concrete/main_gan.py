from generator_gan import TableDataGenerator
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

data = pd.read_csv("data/concrete_data.csv")
print(data)
print(data.dtypes)

train, test = data[:800].to_numpy(), data[800:].to_numpy()
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

all_cols = list(data.columns)
numeric_idx = [all_cols.index(col) for col in data.select_dtypes("number").columns]
cat_idx = [all_cols.index(col) for col in data.select_dtypes("object").columns]

generator = TableDataGenerator(train_scaled, cat_idx, numeric_idx, all_cols, optuna_timeout_seconds=40, n_optuna_trials_per_column=5000, train_epochs=20_000)

for thresh in [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]:
    generated = generator.generate(3_000, thresh=thresh)
    if generated.shape[0] > 0:
        print(f"Sampling with thresh > {thresh}")
        break
generated = scaler.inverse_transform(generated)
generated = pd.DataFrame(generated, columns=all_cols)
print("Generated: ")
print(generated)
generated = generated.to_numpy()

normal_cat = CatBoostRegressor(verbose=0, random_state=42)
normal_cat.fit(train[:, :-1], train[:, -1])
preds_test = normal_cat.predict(test[:, :-1])

scores = [[
    "Реальные данные",
    round(mean_absolute_percentage_error(test[:, -1], preds_test)*100, 1),
    round(mean_absolute_error(test[:, -1], preds_test)*100, 1),
    round(r2_score(test[:, -1], preds_test)*100, 1),
]]

fake_cat = CatBoostRegressor(verbose=0, random_state=42)
fake_cat.fit(generated[:, :-1], generated[:, -1])
preds_fake = fake_cat.predict(test[:, :-1])
scores.append([
    "Сгенерированные данные",
    round(mean_absolute_percentage_error(test[:, -1], preds_fake)*100, 1),
    round(mean_absolute_error(test[:, -1], preds_fake)*100, 1),
    round(r2_score(test[:, -1], preds_fake)*100, 1),
])

concat = np.row_stack([train, generated])

combo_cat = CatBoostRegressor(verbose=0, random_state=42)
combo_cat.fit(concat[:, :-1], concat[:, -1])
preds_combo = combo_cat.predict(test[:, :-1])
scores.append([
    "Реальные + сгенерированные данные",
    round(mean_absolute_percentage_error(test[:, -1], preds_combo)*100, 1),
    round(mean_absolute_error(test[:, -1], preds_combo)*100, 1),
    round(r2_score(test[:, -1], preds_combo)*100, 1),
])

scores = pd.DataFrame(scores, columns=["Тип данных", "MAPE, %", "MAE", "R2"])
print(scores)