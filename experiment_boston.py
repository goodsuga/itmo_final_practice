import pandas as pd
from my_generator import MyGenerator
from gan_generator import GanGenerator
from vae_generator import VaeGenerator
import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score
from tqdm import tqdm

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

E = tqdm(range(20000), total=20000)
for epoch in E:
    opt.zero_grad()
    o, c, e = model(data.iloc[:100])
    loss1 = ((o - e)**2).mean()
    loss2 = ((c - 1)**2).mean()
    loss = loss1 + loss2
    o1, c1, e1 = model(generate_noise_table(100))
    
    idx = np.arange(e.shape[0])
    RNG.shuffle(idx)
    
    loss3 = ((o1 - e[idx])**2).mean()
    loss4 = ((c1 - 0)**2).mean()
    loss2 = loss3 + loss4
    loss = loss + loss2
    loss.backward()
    opt.step()
    E.set_postfix({"loss": round(torch.sqrt(loss).item(), 3)})

model.eval()
with torch.no_grad():
    o, c, _ = model(generate_noise_table(30000))
    o1 = model.decode_output(o[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
    o = model.decode_output(o.to("cuda:0"))
    o2, c, _ = model(data.iloc[:100])
    o3 = model.decode_output(o2[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
    o2 = model.decode_output(o2.to("cuda:0"))

data[cat_features] = data[cat_features].astype(str)
o[cat_features] = o[cat_features].astype(str)
o1[cat_features] = o[cat_features].astype(str)
o2[cat_features] = o[cat_features].astype(str)
o3[cat_features] = o[cat_features].astype(str)

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(data.iloc[:100].drop(columns=["medv"]), data.iloc[:100]["medv"])
preds_raw = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o.drop(columns=["medv"]), o["medv"])
preds_generated1 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o1.drop(columns=["medv"]), o1["medv"])
preds_generated2 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o2.drop(columns=["medv"]), o2["medv"])
preds_generated3 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o3.drop(columns=["medv"]), o3["medv"])
preds_generated4 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o, data.iloc[:100]])["medv"]
)
preds_combo = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o1, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o1, data.iloc[:100]])["medv"]
)
preds_combo1 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o2, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o2, data.iloc[:100]])["medv"]
)
preds_combo2 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o3, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o3, data.iloc[:100]])["medv"]
)
preds_combo3 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

res = pd.DataFrame([
    {"name": "raw", "mae": mae(data.iloc[100:]["medv"], preds_raw), "mape": mape(data.iloc[100:]["medv"], preds_raw), "r2": r2_score(data.iloc[100:]["medv"], preds_raw)},
    {"name": "gen-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_generated1), "mape": mape(data.iloc[100:]["medv"], preds_generated1), "r2": r2_score(data.iloc[100:]["medv"], preds_generated1)},
    {"name": "gen-best-noise", "mae": mae(data.iloc[100:]["medv"], preds_generated2), "mape": mape(data.iloc[100:]["medv"], preds_generated2), "r2": r2_score(data.iloc[100:]["medv"], preds_generated2)},
    {"name": "gen-all-real", "mae": mae(data.iloc[100:]["medv"], preds_generated3), "mape": mape(data.iloc[100:]["medv"], preds_generated3), "r2": r2_score(data.iloc[100:]["medv"], preds_generated3)},
    {"name": "gen-best-real", "mae": mae(data.iloc[100:]["medv"], preds_generated4), "mape": mape(data.iloc[100:]["medv"], preds_generated4), "r2": r2_score(data.iloc[100:]["medv"], preds_generated4)},
    {"name": "combo-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_combo), "mape": mape(data.iloc[100:]["medv"], preds_combo), "r2": r2_score(data.iloc[100:]["medv"], preds_combo)},
    {"name": "combo-best-noise", "mae": mae(data.iloc[100:]["medv"], preds_combo1), "mape": mape(data.iloc[100:]["medv"], preds_combo1), "r2": r2_score(data.iloc[100:]["medv"], preds_combo1)},
    {"name": "combo-all-real", "mae": mae(data.iloc[100:]["medv"], preds_combo2), "mape": mape(data.iloc[100:]["medv"], preds_combo2), "r2": r2_score(data.iloc[100:]["medv"], preds_combo2)},
    {"name": "combo-best-real", "mae": mae(data.iloc[100:]["medv"], preds_combo3), "mape": mape(data.iloc[100:]["medv"], preds_combo3), "r2": r2_score(data.iloc[100:]["medv"], preds_combo3)},
])
print(res.sort_values(by="r2", ascending=False))


# GAN

gan_model = GanGenerator(data, cat_features, 3).to("cuda:0")
opt_creator = torch.optim.Adam(gan_model.creator.parameters(), lr=1e-4)
opt_critic = torch.optim.Adam(gan_model.critic.parameters(), lr=1e-3)

data[cat_features] = data[cat_features].astype(float).astype(int)

E = tqdm(range(20000), total=20000)
loss_critic = None
loss_creator = None
for epoch in E:
    if epoch % 2 == 0:
        opt_creator.zero_grad()
        creation = gan_model(data)
        is_real = gan_model.critic(creation)
        loss_creator = ((is_real - 1)**2).mean()
        loss_creator.backward()
        opt_creator.step()
    else:
        opt_critic.zero_grad()
        creation = gan_model(data)
        is_real = gan_model.critic(creation)
        encoded = gan_model.encode(data.iloc[:100])
        is_real2 = gan_model.critic(encoded.to("cuda:0"))
        loss_critic = ((is_real2 - 1)**2).mean() + ((is_real - 0)**2).mean()
        loss_critic.backward()
        opt_critic.step()
    if loss_critic is not None and loss_creator is not None:
        E.set_postfix({"critic": round(loss_critic.item(), 4), "creator": round(loss_creator.item(), 4)})

model.eval()
with torch.no_grad():
    o = gan_model.forward(generate_noise_table(30000))
    c = gan_model.critic(o)
    o1 = gan_model.decode_output(o[torch.argsort(c.squeeze(), descending=True)[:300]].to("cuda:0"))
    o = gan_model.decode_output(o)

data[cat_features] = data[cat_features].astype(str)
o[cat_features] = o[cat_features].astype(str)
o1[cat_features] = o[cat_features].astype(str)

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(data.iloc[:100].drop(columns=["medv"]), data.iloc[:100]["medv"])
preds_raw = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o.drop(columns=["medv"]), o["medv"])
preds_generated1 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o1.drop(columns=["medv"]), o1["medv"])
preds_generated2 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o, data.iloc[:100]])["medv"]
)
preds_combo = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o1, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o1, data.iloc[:100]])["medv"]
)
preds_combo1 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

res_gan = pd.DataFrame([
    {"name": "raw", "mae": mae(data.iloc[100:]["medv"], preds_raw), "mape": mape(data.iloc[100:]["medv"], preds_raw), "r2": r2_score(data.iloc[100:]["medv"], preds_raw)},
    {"name": "gen-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_generated1), "mape": mape(data.iloc[100:]["medv"], preds_generated1), "r2": r2_score(data.iloc[100:]["medv"], preds_generated1)},
    {"name": "gen-best-noise", "mae": mae(data.iloc[100:]["medv"], preds_generated2), "mape": mape(data.iloc[100:]["medv"], preds_generated2), "r2": r2_score(data.iloc[100:]["medv"], preds_generated2)},
    {"name": "combo-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_combo), "mape": mape(data.iloc[100:]["medv"], preds_combo), "r2": r2_score(data.iloc[100:]["medv"], preds_combo)},
    {"name": "combo-best-noise", "mae": mae(data.iloc[100:]["medv"], preds_combo1), "mape": mape(data.iloc[100:]["medv"], preds_combo1), "r2": r2_score(data.iloc[100:]["medv"], preds_combo1)}
])
print(res_gan.sort_values(by="r2", ascending=False))

# VAE

vae_model = VaeGenerator(data, cat_features, 3).to("cuda:0")
opt_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-4)

data[cat_features] = data[cat_features].astype(float).astype(int)

E = tqdm(range(20000), total=20000)
for epoch in E:
    opt_vae.zero_grad()
    encoded = vae_model.encode(data.iloc[:100]).to("cuda:0")
    creation = vae_model(data.iloc[:100])
    loss = ((creation - encoded)**2).mean() + vae_model.kl
    loss.backward()
    opt_vae.step
    E.set_postfix({"loss": round(loss.item(), 4)})

model.eval()
with torch.no_grad():
    o = vae_model.forward(generate_noise_table(400))
    o = gan_model.decode_output(o)

data[cat_features] = data[cat_features].astype(str)
o[cat_features] = o[cat_features].astype(str)

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(data.iloc[:100].drop(columns=["medv"]), data.iloc[:100]["medv"])
preds_raw = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(o.drop(columns=["medv"]), o["medv"])
preds_generated1 = reg.predict(data.iloc[100:].drop(columns=["medv"]))

reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
reg.fit(
    pd.concat([o, data.iloc[:100]]).drop(columns=["medv"]),
    pd.concat([o, data.iloc[:100]])["medv"]
)
preds_combo = reg.predict(data.iloc[100:].drop(columns=["medv"]))

res_vae = pd.DataFrame([
    {"name": "raw", "mae": mae(data.iloc[100:]["medv"], preds_raw), "mape": mape(data.iloc[100:]["medv"], preds_raw), "r2": r2_score(data.iloc[100:]["medv"], preds_raw)},
    {"name": "gen-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_generated1), "mape": mape(data.iloc[100:]["medv"], preds_generated1), "r2": r2_score(data.iloc[100:]["medv"], preds_generated1)},
    {"name": "combo-all-noise", "mae": mae(data.iloc[100:]["medv"], preds_combo), "mape": mape(data.iloc[100:]["medv"], preds_combo), "r2": r2_score(data.iloc[100:]["medv"], preds_combo)},
])
print(res_vae.sort_values(by="r2", ascending=False))

