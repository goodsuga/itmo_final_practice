from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from kan import KAN
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Generator(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features, 356),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.mu = torch.nn.Linear(356, 356)
        self.sigma = torch.nn.Linear(356, 356)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(356, 356),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(356, in_features)
        )
        
        self.kl = 0.0

    def forward(self, x):
        samples = torch.rand((x.shape[0], 356)).to("cuda:0" if torch.cuda.is_available() else "cpu")
        base = self.encoder(x)
        mu = self.mu(base)
        sigma = self.sigma(base)
        #self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        #print(f"{self.kl=}: {(sigma**2 + mu**2).sum()=}")
        self.kl = (mu**2 + sigma**2).mean()
        samples = samples * self.mu(base) + self.sigma(base)
        return self.decoder(samples)


class TableDataGenerator:
    def __init__(
        self,
        table: np.ndarray,
        categorical_indecies: List[int],
        numeric_indecies: List[int],
        column_names: List[str],
        sample_count_during_pick: Optional[int] = 2000,
        n_distributions_per_column: Optional[int] = 3,
        plot_report_dir: Optional[Path] = Path("generator_plots"),
        n_optuna_trials_per_column: Optional[int] = 2000,
        optuna_timeout_seconds: Optional[int] = 120,
        device_str: Optional[str] = "cuda:0" if torch.cuda.is_available() else "cpu",
        train_epochs: Optional[int] = 1000,
    ):
        # Несколько важных ожиданий:
        # 1. Числовые данные должны быть нормализованы от 0 до 1 или от -1 до 1
        # 2. Категориальные признаки должны быть закодированы порядково
        # (категория заменена ее индексом)
        # 3. В данных не должно быть пропусков

        self.sample_count_during_pick = sample_count_during_pick
        self.n_distributions_per_column = n_distributions_per_column
        self.plot_report_dir = plot_report_dir
        self.plot_report_dir.mkdir(parents=True, exist_ok=True)
        self.n_optuna_trials_per_column = n_optuna_trials_per_column
        self.optuna_timeout_seconds = optuna_timeout_seconds
        self.train_epochs = train_epochs

        self.total_in_features = table.shape[1]

        self.categorical_indecies = list(categorical_indecies)
        self.numeric_indecies = list(numeric_indecies)
        self.column_names = column_names

        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        if self.categorical_indecies:
            self.one_hot_encoder.fit(table[:, self.categorical_indecies])

        if self.categorical_indecies and self.numeric_indecies:
            temporary_repr = np.column_stack(
                [
                    table[:, self.numeric_indecies],
                    self.one_hot_encoder.transform(table[:, self.categorical_indecies]),
                ]
            )
        elif self.numeric_indecies:
            temporary_repr = table
        elif self.categorical_indecies:
            temporary_repr = self.one_hot_encoder.transform(table)
        else:
            raise ValueError(
                "Both numeric indecies and categorical indecies are empty lists!"
            )

        print(f"{temporary_repr.shape=}")

        temporary_repr = torch.as_tensor(temporary_repr).float().to(device_str)

        self.generator = Generator(in_features=temporary_repr.shape[1]).to(device_str)
        self.optimizer_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=1e-4
        )

        self.in_features_full = temporary_repr.shape[1]

        print(
            f"Count of VAE generator params: {sum(param.numel() for param in self.generator.parameters())}"
        )
        
        print_loss = False
        train_range = tqdm(
            range(self.train_epochs), total=self.train_epochs, desc="Обучаю VAE"
        )

        for epoch in train_range:
            def closure():
                
                # Тренируем генератор
                self.optimizer_generator.zero_grad()
                
                samples = self.generator(temporary_repr)
                loss = ((samples - temporary_repr)**2).mean() + self.generator.kl

                if print_loss:
                    train_range.set_postfix(
                        {
                            "epoch": epoch,
                            "loss": round(loss.item(), 4)
                        }
                    )

                if loss.requires_grad:
                    loss.backward()
                return loss

            self.optimizer_generator.step(closure)

            with torch.no_grad():
                print_loss = True
                closure()
                print_loss = False

    def generate(
        self, n
    ):
        # n - сырая генерация. Итоговый набор может быть и пустым
        self.generator.to("cpu").eval()
        
        with torch.no_grad():
            to_return = self.generator.decoder(torch.rand((n, 356))).detach().cpu().numpy()

        if self.categorical_indecies:
            rebuilt_return = np.zeros((to_return.shape[0], self.total_in_features))
            
            categorical_part = to_return[:, len(self.numeric_indecies):]
            current_pos = 0
            for i, categories in enumerate(self.one_hot_encoder.categories_):
                n_categories = len(categories)
                sampled_categories = np.argmax(categorical_part[:, current_pos:current_pos+n_categories], axis=1)
                assert len(sampled_categories.shape) == 1, f"sampled categories should have shape (n, ), whereas found {sampled_categories.shape}"
                rebuilt_return[:, self.categorical_indecies[i]] = sampled_categories.squeeze()
                current_pos += n_categories
            assert current_pos == sum(len(cats) for cats in self.one_hot_encoder.categories_)
            
            for i in range(len(self.numeric_indecies)):
                rebuilt_return[:, self.numeric_indecies[i]] = to_return[:, i]

            to_return = rebuilt_return

        return to_return
