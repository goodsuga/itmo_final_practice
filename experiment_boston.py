import pandas as pd
from my_generator import MyGenerator
import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score

RNG = np.random.default_rng(42)

data = pd.read_csv("BostonHousing.csv")
cat_features = ["chas", "rad"]
for col in cat_features:
    data[col] = OrdinalEncoder().fit_transform(data[col].fillna("NaN").to_numpy().reshape((-1, 1)))

model = MyGenerator(data, cat_features, 3).to("cuda:0")
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


def generate_noise_table(samples):
    cols = {}
    for col in model.stats:
        if "nunique" in model.stats[col]:
            cols[col] = RNG.integers(0, model.stats[col]["nunique"], samples)
        else:
            cols[col] = RNG.uniform(model.stats[col]["min"], model.stats[col]["max"], samples)
    return pd.DataFrame(cols)
    
for epoch in range(500):
    opt.zero_grad()
    o, c, e = model(data.iloc[:400])
    loss1 = ((o - e)**2).mean()
    loss2 = ((c - 1)**2).mean()
    loss = loss1 + loss2
    o1, c1, e1 = model(generate_noise_table(400))
    
    idx = np.arange(e.shape[0])
    RNG.shuffle(idx)
    
    loss3 = ((o1 - e[idx])**2).mean()
    loss4 = ((c1 - 0)**2).mean()
    loss2 = loss3 + loss4
    loss = loss + loss2
    loss.backward()
    opt.step()
    print(f"{epoch=}: {torch.sqrt(loss).item()=:.4f}")

model.eval()
with torch.no_grad():
    o, _, _ = model(generate_noise_table(400))
    o = model.decode_output(o.to("cuda:0"))

o.loc[:, cat_features] = o[cat_features].astype(str)
data.loc[:, cat_features] = data[cat_features].astype(str)

print("DATA:")
print(data)
print("O:")
print(o)

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(data.iloc[:400].drop(columns=["medv"]), data.iloc[:400]["medv"])
preds_raw = reg.predict(data.iloc[400:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o.drop(columns=["medv"]), o["medv"])
preds_generated = reg.predict(data.iloc[400:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o, data.iloc[:400]]).drop(columns=["medv"]),
    pd.concat([o, data.iloc[:400]])["medv"]
)
preds_combo = reg.predict(data.iloc[400:].drop(columns=["medv"]))

res = pd.DataFrame([
    {"name": "raw", "mae": mae(data.iloc[400:]["medv"], preds_raw), "mape": mape(data.iloc[400:]["medv"], preds_raw), "r2": r2_score(data.iloc[400:]["medv"], preds_raw)},
    {"name": "gen", "mae": mae(data.iloc[400:]["medv"], preds_generated), "mape": mape(data.iloc[400:]["medv"], preds_generated), "r2": r2_score(data.iloc[400:]["medv"], preds_generated)},
    {"name": "combo", "mae": mae(data.iloc[400:]["medv"], preds_combo), "mape": mape(data.iloc[400:]["medv"], preds_combo), "r2": r2_score(data.iloc[400:]["medv"], preds_combo)}
])
print(res)

    

