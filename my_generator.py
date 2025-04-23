import pandas as pd
import torch
from typing import List
from tqdm import tqdm
import numpy as np


class MyGenerator(torch.nn.Module):
    def __init__(self, dataframe: pd.DataFrame, cat_features: List[str], desired_embedding_size: int):
        super().__init__()
        # Кодируем входное пространство
        self.cat_features = cat_features
        self.desired_embedding_size = desired_embedding_size
        self.embedded_size = 0
        self.embedding_scheme = {}
        self.stats = {}
        self.embedding_layers = {}
        
        for i, feature in enumerate(dataframe.columns):
            col = dataframe[feature]
            if feature in cat_features:
                self.stats[feature] = {"nunique": col.nunique()}
                self.embedding_scheme[feature] = (self.embedded_size, self.embedded_size + desired_embedding_size)
                self.embedded_size += desired_embedding_size
                self.embedding_layers[feature] = torch.nn.Embedding(self.stats[feature]["nunique"], desired_embedding_size)
            else:
                self.stats[feature] = {"min": col.min(), "max": col.max()}
                self.embedding_scheme[feature] = (self.embedded_size, self.embedded_size + 1)
                self.embedded_size += 1
                self.embedding_layers[feature] = lambda x: x #torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.PReLU())
            
        self.embedding_layers = torch.nn.ParameterDict(self.embedding_layers)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.embedded_size, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, self.embedded_size + 1)
        )
        
        self.pdist = torch.nn.PairwiseDistance()
        
    def forward(self, table: pd.DataFrame):
        encoded_input = torch.zeros((table.shape[0], self.embedded_size))
        for col in table.columns:
            scheme = self.embedding_scheme[col]
            values = table[col].to_numpy().reshape((-1, 1))
            if col not in self.cat_features:
                values = (values - self.stats[col]["min"]) / (self.stats[col]["max"] - self.stats[col]["min"])
            values = torch.as_tensor(values, dtype=torch.int64 if col in self.cat_features else torch.float32)
            if col in self.cat_features:
                values = values[:, 0]
            encoded_input[:, scheme[0]:scheme[1]] = self.embedding_layers[col](values.to("cuda:0"))
        full_out = self.encoder(encoded_input.to("cuda:0"))
        out = full_out[:, :-1].cpu()
        cls = full_out[:, -1].cpu()
        return out, cls, encoded_input
    
    @torch.no_grad()
    def decode_output(self, out):
        cols = {}
        for col in self.embedding_scheme:
            scheme = self.embedding_scheme[col]
            col_out = out[:, scheme[0]:scheme[1]]
            if col in self.cat_features:
                embeddings = self.embedding_layers[col].weight.data
                dist = np.array([
                    int(torch.argmax(((embeddings - col_out[i])**2).mean(dim=-1)).item())
                    for i in range(col_out.shape[0])
                ])
                cols[col] = dist
            else:
                cols[col] = (col_out * (self.stats[col]["max"] - self.stats[col]["min"]) + self.stats[col]["min"]).cpu().numpy().flatten()
        return pd.DataFrame(cols)

if __name__ == "__main__":
    table = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [1, 0, 2, 0, 1, 2],
        "C": [1, 2, 3, 4, 5, 6]
    })
    generator = MyGenerator(table, cat_features=["B"], desired_embedding_size=3).to("cuda:0")
    opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    for epoch in tqdm(list(range(100))):
        opt.zero_grad()
        o, c, e = generator(table)
        loss1 = ((o - e)**2).mean()
        loss2 = ((c - 1)**2).mean()
        loss = loss1 + loss2
        loss.backward()
        opt.step()
        print(f"{loss1.item()=}; {loss2.item()=}; {loss.item()=}")
    
    o, _, _ = generator(table)
    print(generator.decode_output(o.to("cuda:0")))

        
        
        
            
