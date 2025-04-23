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
    # def __init__(self, in_features: int):
    #     super().__init__()
    #     self.model = KAN([in_features, in_features], k=8, base_fun=torch.nn.functional.tanh, auto_save=False).speed()
    
    # def forward(self, x):
    #     return self.model(x)
    def __init__(self, in_features: int):
        super().__init__()
        # self.n_layers = 15
        # self.layers = torch.nn.ParameterList([
        #     torch.nn.Linear(in_features if i == 0 else 356, in_features if i == self.n_layers -1 else 356)
        #     for i in range(self.n_layers)
        # ])
        # #self.drop = torch.nn.Dropout(0.1)
        # self.relu = torch.nn.ReLU()
        # self.ln = torch.nn.LayerNorm(356)
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 356),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(356, 356),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(356, in_features),
        )

    def forward(self, x):
        return self.model(x)
        #out = self.ln(self.relu(self.layers[0](x)))
        #for i in range(1, self.n_layers-1):
        #    next_out = self.ln(self.relu(self.layers[i](out)))
        #    out = next_out + out
        #return self.layers[-1](out)


class Discriminator(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        #self.model = KAN([in_features, 3, 2, 1], auto_save=False).speed()
        self.in_features = in_features
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.nn.functional.sigmoid(self.model(x))

last_known_gen_loss = 0.0
last_known_disc_loss = 0.0

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
        real_label = torch.ones((temporary_repr.shape[0], 1)).to(device_str)
        fake_label = torch.zeros((temporary_repr.shape[0], 1)).to(device_str)

        self.discriminator = Discriminator(in_features=temporary_repr.shape[1]).to(
            device_str
        )
        self.generator = Generator(in_features=temporary_repr.shape[1]).to(device_str)
        self.optimizer_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(), lr=1e-4
        )
        self.optimizer_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=1e-4
        )

        self.in_features_full = temporary_repr.shape[1]

        print(
            f"Count of generator params: {sum(param.numel() for param in self.generator.parameters())}"
        )
        print(
            f"Count of discriminator params: {sum(param.numel() for param in self.discriminator.parameters())}"
        )

        print_loss = False
        train_range = tqdm(
            range(self.train_epochs), total=self.train_epochs, desc="Обучаю GAN"
        )

        for epoch in train_range:
            def closure():
                global last_known_gen_loss, last_known_disc_loss
                if epoch % 2 == 0:
                    # Тренируем дискриминатор
                    self.optimizer_discriminator.zero_grad()
                    samples = self.generator(torch.rand((temporary_repr.shape[0], self.in_features_full)).to(device_str))

                    is_real = self.discriminator(temporary_repr)
                    discriminator_loss = torch.nn.functional.binary_cross_entropy(
                        is_real, real_label
                    )

                    is_real_on_fakes = self.discriminator(samples)
                    discriminator_loss = (
                        discriminator_loss
                        + torch.nn.functional.binary_cross_entropy(
                            is_real_on_fakes, fake_label
                        )
                    )

                    if print_loss:
                        last_known_disc_loss = round(discriminator_loss.item(), 4)
                        train_range.set_postfix(
                            {
                                "epoch": epoch,
                                "discriminator loss": last_known_disc_loss,
                                "generator loss": last_known_gen_loss
                            }
                        )

                    if discriminator_loss.requires_grad:
                        discriminator_loss.backward()
                    return discriminator_loss
                else:
                    # Тренируем генератор
                    self.optimizer_generator.zero_grad()
                    samples = self.generator(torch.rand((temporary_repr.shape[0], self.in_features_full)).to(device_str))

                    is_real_on_fakes = self.discriminator(samples)
                    # здесь real_label вместо fake_label - нам нужно чтобы дискриминатор признал все сэмплы
                    # правдой
                    generator_loss = torch.nn.functional.binary_cross_entropy(
                        is_real_on_fakes, real_label
                    )

                    if print_loss:
                        last_known_gen_loss = round(generator_loss.item(), 4)
                        train_range.set_postfix(
                            {
                                "epoch": epoch,
                                "discriminator loss": last_known_disc_loss,
                                "generator loss": last_known_gen_loss
                            }
                        )

                    if generator_loss.requires_grad:
                        generator_loss.backward()
                    return generator_loss

            if epoch % 2 == 0:
                self.optimizer_discriminator.step(closure)
            else:
                self.optimizer_generator.step(closure)

            with torch.no_grad():
                print_loss = True
                closure()
                print_loss = False

    def generate(
        self, n, pick_real: Optional[bool] = True, thresh: Optional[float] = 0.5
    ):
        # n - сырая генерация. Итоговый набор может быть и пустым
        self.generator.to("cpu").eval()
        self.discriminator.to("cpu").eval()
        
        with torch.no_grad():
            samples = self.generator(torch.rand((n, self.in_features_full)))
            is_real = self.discriminator(samples)
            samples = samples.detach().cpu().numpy()
            is_real = is_real.detach().cpu().numpy()

        if pick_real:
            to_return = samples[is_real.squeeze() > thresh, :]
        else:
            to_return = samples

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
