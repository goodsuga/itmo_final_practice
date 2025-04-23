import pandas as pd
from my_generator import MyGenerator
import torch
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

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
    
for epoch in range(10000):
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
    print(f"{loss.item()=:.4f}")
