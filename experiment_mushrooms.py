import pandas as pd
from my_generator import MyGenerator
from gan_generator import GanGenerator
from vae_generator import VaeGenerator
import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score as acc, f1_score as f1, roc_auc_score as ras
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4

RNG = np.random.default_rng(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

def run_mushrooms():

    outdir = Path("mushrooms_results")
    out_my = outdir / "my"
    out_my.mkdir(exist_ok=True, parents=True)
    out_gan = outdir / "gan"
    out_gan.mkdir(exist_ok=True, parents=True)
    out_vae = outdir / "vae"
    out_vae.mkdir(exist_ok=True, parents=True)


    data = pd.read_csv("data/mushrooms.csv")
    cat_features = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat"
    ]
    TARGET = "class"
    for col in cat_features:
        data[col] = OrdinalEncoder().fit_transform(data[col].fillna("?").to_numpy().reshape((-1, 1)))
    
    data = data.sample(frac=1.0).reset_index(drop=True)
    
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

    E = tqdm(range(30000), total=30000)
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
        o, c, _ = model(generate_noise_table(500))
        o1 = model.decode_output(o[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
        o = model.decode_output(o.to("cuda:0"))
        o2, c, _ = model(data.iloc[:300])
        o3 = model.decode_output(o2[torch.argsort(c, descending=True)[:300]].to("cuda:0"))
        o2 = model.decode_output(o2.to("cuda:0"))

    short_cat = [c for c in cat_features if c != TARGET]
    data[short_cat] = data[short_cat].astype(str)
    data[TARGET] = data[TARGET].apply(lambda x: int(float(x)))
    o[short_cat] = o[short_cat].astype(str)
    o1[short_cat] = o[short_cat].astype(str)
    o2[short_cat] = o[short_cat].astype(str)
    o3[short_cat] = o[short_cat].astype(str)
    
    reg = CatBoostClassifier(allow_const_label=True,cat_features=[c for c in cat_features if c != TARGET], verbose=0, depth=3)
    
    reg.fit(data.iloc[:300].drop(columns=[TARGET]), data.iloc[:300][TARGET])
    preds_raw = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o.drop(columns=[TARGET]), o[TARGET])
    preds_generated1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o1.drop(columns=[TARGET]), o1[TARGET])
    preds_generated2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o2.drop(columns=[TARGET]), o2[TARGET])
    preds_generated3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o3.drop(columns=[TARGET]), o3[TARGET])
    preds_generated4 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o, data.iloc[:300]])[TARGET]
    )
    preds_combo = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o1, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o1, data.iloc[:300]])[TARGET]
    )
    preds_combo1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o2, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o2, data.iloc[:300]])[TARGET]
    )
    preds_combo2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o3, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o3, data.iloc[:300]])[TARGET]
    )
    preds_combo3 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    res = pd.DataFrame([
        {"name": "raw", "acc": acc(data.iloc[300:][TARGET], preds_raw), "f1": f1(data.iloc[300:][TARGET], preds_raw), "ras": ras(data.iloc[300:][TARGET], preds_raw)},
        {"name": "gen-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_generated1), "f1": f1(data.iloc[300:][TARGET], preds_generated1), "ras": ras(data.iloc[300:][TARGET], preds_generated1)},
        {"name": "gen-best-noise", "acc": acc(data.iloc[300:][TARGET], preds_generated2), "f1": f1(data.iloc[300:][TARGET], preds_generated2), "ras": ras(data.iloc[300:][TARGET], preds_generated2)},
        {"name": "gen-all-real", "acc": acc(data.iloc[300:][TARGET], preds_generated3), "f1": f1(data.iloc[300:][TARGET], preds_generated3), "ras": ras(data.iloc[300:][TARGET], preds_generated3)},
        {"name": "gen-best-real", "acc": acc(data.iloc[300:][TARGET], preds_generated4), "f1": f1(data.iloc[300:][TARGET], preds_generated4), "ras": ras(data.iloc[300:][TARGET], preds_generated4)},
        {"name": "combo-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_combo), "f1": f1(data.iloc[300:][TARGET], preds_combo), "ras": ras(data.iloc[300:][TARGET], preds_combo)},
        {"name": "combo-best-noise", "acc": acc(data.iloc[300:][TARGET], preds_combo1), "f1": f1(data.iloc[300:][TARGET], preds_combo1), "ras": ras(data.iloc[300:][TARGET], preds_combo1)},
        {"name": "combo-all-real", "acc": acc(data.iloc[300:][TARGET], preds_combo2), "f1": f1(data.iloc[300:][TARGET], preds_combo2), "ras": ras(data.iloc[300:][TARGET], preds_combo2)},
        {"name": "combo-best-real", "acc": acc(data.iloc[300:][TARGET], preds_combo3), "f1": f1(data.iloc[300:][TARGET], preds_combo3), "ras": ras(data.iloc[300:][TARGET], preds_combo3)},
    ])
    res.sort_values(by="ras", ascending=False).to_csv(out_my / (str(uuid4()) + ".csv"), index=False)


    # GAN

    gan_model = GanGenerator(data, cat_features, 3).to("cuda:0")
    opt_creator = torch.optim.Adam(gan_model.creator.parameters(), lr=1e-3)
    opt_critic = torch.optim.Adam(gan_model.critic.parameters(), lr=1e-4)

    data[cat_features] = data[cat_features].astype(float).astype(int)

    E = tqdm(range(10000), total=10000)
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
            encoded = gan_model.encode(data.iloc[:300])
            is_real2 = gan_model.critic(encoded.to("cuda:0"))
            loss_critic = ((is_real2 - 1)**2).mean() + ((is_real - 0)**2).mean()
            loss_critic.backward()
            opt_critic.step()
        if loss_critic is not None and loss_creator is not None:
            E.set_postfix({"critic": round(loss_critic.item(), 4), "creator": round(loss_creator.item(), 4)})

    model.eval()
    with torch.no_grad():
        o = gan_model.forward(generate_noise_table(500))
        c = gan_model.critic(o)
        o1 = gan_model.decode_output(o[torch.argsort(c.squeeze(), descending=True)[:300]].to("cuda:0"))
        o = gan_model.decode_output(o)

    data[short_cat] = data[short_cat].astype(str)
    data[TARGET] = data[TARGET].apply(lambda x: int(float(x)))
    o[short_cat] = o[short_cat].astype(str)
    o1[short_cat] = o[short_cat].astype(str)

    
    reg.fit(data.iloc[:300].drop(columns=[TARGET]), data.iloc[:300][TARGET])
    preds_raw = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o.drop(columns=[TARGET]), o[TARGET])
    preds_generated1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o1.drop(columns=[TARGET]), o1[TARGET])
    preds_generated2 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o, data.iloc[:300]])[TARGET]
    )
    preds_combo = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o1, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o1, data.iloc[:300]])[TARGET]
    )
    preds_combo1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    res_gan = pd.DataFrame([
        {"name": "raw", "acc": acc(data.iloc[300:][TARGET], preds_raw), "f1": f1(data.iloc[300:][TARGET], preds_raw), "ras": ras(data.iloc[300:][TARGET], preds_raw)},
        {"name": "gen-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_generated1), "f1": f1(data.iloc[300:][TARGET], preds_generated1), "ras": ras(data.iloc[300:][TARGET], preds_generated1)},
        {"name": "gen-best-noise", "acc": acc(data.iloc[300:][TARGET], preds_generated2), "f1": f1(data.iloc[300:][TARGET], preds_generated2), "ras": ras(data.iloc[300:][TARGET], preds_generated2)},
        {"name": "combo-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_combo), "f1": f1(data.iloc[300:][TARGET], preds_combo), "ras": ras(data.iloc[300:][TARGET], preds_combo)},
        {"name": "combo-best-noise", "acc": acc(data.iloc[300:][TARGET], preds_combo1), "f1": f1(data.iloc[300:][TARGET], preds_combo1), "ras": ras(data.iloc[300:][TARGET], preds_combo1)}
    ])
    res_gan.sort_values(by="ras", ascending=False).to_csv(out_gan / (str(uuid4()) + ".csv"), index=False)

    # VAE

    vae_model = VaeGenerator(data, cat_features, 3).to("cuda:0")
    opt_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-4)

    data[cat_features] = data[cat_features].astype(float).astype(int)

    E = tqdm(range(10000), total=10000)
    for epoch in E:
        opt_vae.zero_grad()
        encoded = vae_model.encode(data.iloc[:300]).to("cuda:0")
        creation = vae_model(data.iloc[:300])
        loss = ((creation - encoded)**2).mean() + vae_model.kl
        loss.backward()
        opt_vae.step
        E.set_postfix({"loss": round(loss.item(), 4)})

    model.eval()
    with torch.no_grad():
        o = vae_model.forward(generate_noise_table(500))
        o = gan_model.decode_output(o)

    data[short_cat] = data[short_cat].astype(str)
    data[TARGET] = data[TARGET].apply(lambda x: int(float(x)))
    o[short_cat] = o[short_cat].astype(str)

    
    reg.fit(data.iloc[:300].drop(columns=[TARGET]), data.iloc[:300][TARGET])
    preds_raw = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(o.drop(columns=[TARGET]), o[TARGET])
    preds_generated1 = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    
    reg.fit(
        pd.concat([o, data.iloc[:300]]).drop(columns=[TARGET]),
        pd.concat([o, data.iloc[:300]])[TARGET]
    )
    preds_combo = reg.predict(data.iloc[300:].drop(columns=[TARGET]))

    res_vae = pd.DataFrame([
        {"name": "raw", "acc": acc(data.iloc[300:][TARGET], preds_raw), "f1": f1(data.iloc[300:][TARGET], preds_raw), "ras": ras(data.iloc[300:][TARGET], preds_raw)},
        {"name": "gen-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_generated1), "f1": f1(data.iloc[300:][TARGET], preds_generated1), "ras": ras(data.iloc[300:][TARGET], preds_generated1)},
        {"name": "combo-all-noise", "acc": acc(data.iloc[300:][TARGET], preds_combo), "f1": f1(data.iloc[300:][TARGET], preds_combo), "ras": ras(data.iloc[300:][TARGET], preds_combo)},
    ])
    res_vae.sort_values(by="ras", ascending=False).to_csv(out_vae / (str(uuid4()) + ".csv"), index=False)

