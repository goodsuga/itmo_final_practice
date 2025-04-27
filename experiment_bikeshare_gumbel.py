import pandas as pd
from my_generator_gumbel import MyGumbelGenerator
from my_generator import MyGenerator
import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4

RNG = np.random.default_rng(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

def run_bikeshare_comp():

    outdir = Path("bikeshare_gumbel_results")
    out_my = outdir / "my"
    out_my.mkdir(exist_ok=True, parents=True)
    out_my_gumbel = outdir / "my_gumbel"
    out_my_gumbel.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv("data/bikeshare.csv").drop(columns=["instant", "dteday", "yr", "season"])
    cat_features = ["mnth", "holiday", "weekday", "workingday", "weathersit"]
    TARGET = "cnt"
    for col in cat_features:
        data[col] = OrdinalEncoder().fit_transform(data[col].fillna("NaN").to_numpy().reshape((-1, 1)))

    model = MyGumbelGenerator(data, cat_features, 3).to("cuda:0")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def generate_noise_table(samples):
        cols = {}
        for col in model.stats:
            if "nunique" in model.stats[col]:
                cols[col] = RNG.integers(0, model.stats[col]["nunique"], samples)
            else:
                cols[col] = RNG.uniform(model.stats[col]["min"], model.stats[col]["max"], samples)
        return pd.DataFrame(cols)

    E = tqdm(range(10), total=10)
    for epoch in E:
        opt.zero_grad()
        o, c, e = model(data.iloc[:300])
        loss1 = ((o - e)**2).mean()
        loss2 = ((c - 1)**2).mean()
        loss = loss1 + loss2
        o1, c1, e1 = model(generate_noise_table(300))
        
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
        o2, c, _ = model(data.iloc[:300])
        o3 = model.decode_output(o2[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
        o2 = model.decode_output(o2.to("cuda:0"))

    data[cat_features] = data[cat_features].astype(str)
    o[cat_features] = o[cat_features].astype(str)
    o1[cat_features] = o[cat_features].astype(str)
    o2[cat_features] = o[cat_features].astype(str)
    o3[cat_features] = o[cat_features].astype(str)

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(data.iloc[:300].drop(columns=[TARGET]), data.iloc[:300][TARGET])
    preds_raw = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o.drop(columns=[TARGET]), o[TARGET])
    preds_generated1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o1.drop(columns=[TARGET]), o1[TARGET])
    preds_generated2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o2.drop(columns=[TARGET]), o2[TARGET])
    preds_generated3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o3.drop(columns=[TARGET]), o3[TARGET])
    preds_generated4 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o, data.iloc[:300]])[TARGET]
    )
    preds_combo = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o1, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o1, data.iloc[:300]])[TARGET]
    )
    preds_combo1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o2, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o2, data.iloc[:300]])[TARGET]
    )
    preds_combo2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o3, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o3, data.iloc[:300]])[TARGET]
    )
    preds_combo3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    res = pd.DataFrame([
        {"name": "raw", "mae": mae(data.iloc[300:][TARGET], preds_raw), "mape": mape(data.iloc[300:][TARGET], preds_raw), "r2": r2_score(data.iloc[300:][TARGET], preds_raw)},
        {"name": "gen-all-noise", "mae": mae(data.iloc[300:][TARGET], preds_generated1), "mape": mape(data.iloc[300:][TARGET], preds_generated1), "r2": r2_score(data.iloc[300:][TARGET], preds_generated1)},
        {"name": "gen-best-noise", "mae": mae(data.iloc[300:][TARGET], preds_generated2), "mape": mape(data.iloc[300:][TARGET], preds_generated2), "r2": r2_score(data.iloc[300:][TARGET], preds_generated2)},
        {"name": "gen-all-real", "mae": mae(data.iloc[300:][TARGET], preds_generated3), "mape": mape(data.iloc[300:][TARGET], preds_generated3), "r2": r2_score(data.iloc[300:][TARGET], preds_generated3)},
        {"name": "gen-best-real", "mae": mae(data.iloc[300:][TARGET], preds_generated4), "mape": mape(data.iloc[300:][TARGET], preds_generated4), "r2": r2_score(data.iloc[300:][TARGET], preds_generated4)},
        {"name": "combo-all-noise", "mae": mae(data.iloc[300:][TARGET], preds_combo), "mape": mape(data.iloc[300:][TARGET], preds_combo), "r2": r2_score(data.iloc[300:][TARGET], preds_combo)},
        {"name": "combo-best-noise", "mae": mae(data.iloc[300:][TARGET], preds_combo1), "mape": mape(data.iloc[300:][TARGET], preds_combo1), "r2": r2_score(data.iloc[300:][TARGET], preds_combo1)},
        {"name": "combo-all-real", "mae": mae(data.iloc[300:][TARGET], preds_combo2), "mape": mape(data.iloc[300:][TARGET], preds_combo2), "r2": r2_score(data.iloc[300:][TARGET], preds_combo2)},
        {"name": "combo-best-real", "mae": mae(data.iloc[300:][TARGET], preds_combo3), "mape": mape(data.iloc[300:][TARGET], preds_combo3), "r2": r2_score(data.iloc[300:][TARGET], preds_combo3)},
    ])
    res.sort_values(by="r2", ascending=False).to_csv(out_my_gumbel / (str(uuid4()) + ".csv"), index=False)

    model = MyGenerator(data, cat_features, 3).to("cuda:0")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    data[cat_features] = data[cat_features].astype(float).astype(int)
    
    def generate_noise_table(samples):
        cols = {}
        for col in model.stats:
            if "nunique" in model.stats[col]:
                cols[col] = RNG.integers(0, model.stats[col]["nunique"], samples)
            else:
                cols[col] = RNG.uniform(model.stats[col]["min"], model.stats[col]["max"], samples)
        return pd.DataFrame(cols)

    E = tqdm(range(10000), total=10000)
    for epoch in E:
        opt.zero_grad()
        o, c, e = model(data.iloc[:300])
        loss1 = ((o - e)**2).mean()
        loss2 = ((c - 1)**2).mean()
        loss = loss1 + loss2
        o1, c1, e1 = model(generate_noise_table(300))
        
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
        o2, c, _ = model(data.iloc[:300])
        o3 = model.decode_output(o2[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
        o2 = model.decode_output(o2.to("cuda:0"))

    data[cat_features] = data[cat_features].astype(str)
    o[cat_features] = o[cat_features].astype(str)
    o1[cat_features] = o[cat_features].astype(str)
    o2[cat_features] = o[cat_features].astype(str)
    o3[cat_features] = o[cat_features].astype(str)

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(data.iloc[:300].drop(columns=[TARGET]), data.iloc[:300][TARGET])
    preds_raw = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o.drop(columns=[TARGET]), o[TARGET])
    preds_generated1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o1.drop(columns=[TARGET]), o1[TARGET])
    preds_generated2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o2.drop(columns=[TARGET]), o2[TARGET])
    preds_generated3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(o3.drop(columns=[TARGET]), o3[TARGET])
    preds_generated4 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o, data.iloc[:300]])[TARGET]
    )
    preds_combo = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o1, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o1, data.iloc[:300]])[TARGET]
    )
    preds_combo1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o2, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o2, data.iloc[:300]])[TARGET]
    )
    preds_combo2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    reg = CatBoostRegressor(cat_features=cat_features, verbose=0)
    reg.fit(
        pd.concat([o3, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o3, data.iloc[:300]])[TARGET]
    )
    preds_combo3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    res = pd.DataFrame([
        {"name": "raw", "mae": mae(data.iloc[300:][TARGET], preds_raw), "mape": mape(data.iloc[300:][TARGET], preds_raw), "r2": r2_score(data.iloc[300:][TARGET], preds_raw)},
        {"name": "gen-all-noise", "mae": mae(data.iloc[300:][TARGET], preds_generated1), "mape": mape(data.iloc[300:][TARGET], preds_generated1), "r2": r2_score(data.iloc[300:][TARGET], preds_generated1)},
        {"name": "gen-best-noise", "mae": mae(data.iloc[300:][TARGET], preds_generated2), "mape": mape(data.iloc[300:][TARGET], preds_generated2), "r2": r2_score(data.iloc[300:][TARGET], preds_generated2)},
        {"name": "gen-all-real", "mae": mae(data.iloc[300:][TARGET], preds_generated3), "mape": mape(data.iloc[300:][TARGET], preds_generated3), "r2": r2_score(data.iloc[300:][TARGET], preds_generated3)},
        {"name": "gen-best-real", "mae": mae(data.iloc[300:][TARGET], preds_generated4), "mape": mape(data.iloc[300:][TARGET], preds_generated4), "r2": r2_score(data.iloc[300:][TARGET], preds_generated4)},
        {"name": "combo-all-noise", "mae": mae(data.iloc[300:][TARGET], preds_combo), "mape": mape(data.iloc[300:][TARGET], preds_combo), "r2": r2_score(data.iloc[300:][TARGET], preds_combo)},
        {"name": "combo-best-noise", "mae": mae(data.iloc[300:][TARGET], preds_combo1), "mape": mape(data.iloc[300:][TARGET], preds_combo1), "r2": r2_score(data.iloc[300:][TARGET], preds_combo1)},
        {"name": "combo-all-real", "mae": mae(data.iloc[300:][TARGET], preds_combo2), "mape": mape(data.iloc[300:][TARGET], preds_combo2), "r2": r2_score(data.iloc[300:][TARGET], preds_combo2)},
        {"name": "combo-best-real", "mae": mae(data.iloc[300:][TARGET], preds_combo3), "mape": mape(data.iloc[300:][TARGET], preds_combo3), "r2": r2_score(data.iloc[300:][TARGET], preds_combo3)},
    ])
    res.sort_values(by="r2", ascending=False).to_csv(out_my / (str(uuid4()) + ".csv"), index=False)
