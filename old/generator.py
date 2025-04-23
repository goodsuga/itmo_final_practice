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

class Distribution:
    def __init__(self, dist_type: str, dist_params: dict):
        self.dist_type = dist_type
        self.dist_params = dist_params

    def sample(self, shape: tuple):
        if self.dist_type == "Beta":
            return np.random.beta(
                self.dist_params["alpha"], self.dist_params["beta"], shape
            )
        elif self.dist_type == "Normal":
            return np.random.normal(
                self.dist_params["loc"], self.dist_params["scale"], shape
            )
        elif self.dist_type == "Chisquare":
            return np.random.chisquare(self.dist_params["df"], shape)
        elif self.dist_type == "Exponential":
            return np.random.exponential(self.dist_params["lambda"], shape)
        elif self.dist_type == "Categorical":
            return np.random.choice(
                self.dist_params["choices"], size=shape, p=self.dist_params["probas"]
            )
        else:
            raise KeyError(self.dist_type)


class MultiDistSampler:
    def __init__(self, params):
        self.n_dists = params["n_dists"]
        self.dists = []
        self.probas = []
        for dist_idx in range(self.n_dists):
            dist_type = params[f"dist_{dist_idx}_type"]
            dist_params = {
                key.replace(f"dist_{dist_idx}_", ""): value
                for key, value in params.items()
                if key.startswith(f"dist_{dist_idx}") and not key.endswith("_type")
            }
            self.dists.append(Distribution(dist_type, dist_params))
            self.probas.append(params[f"dist_{dist_idx}_proba"])
        self.probas = np.array(self.probas, dtype=np.float32)
        self.probas_norm = self.probas / self.probas.sum()
        self.choose_from = np.arange(len(self.dists))

    def sample(self, shape):
        samples = np.column_stack(
            [self.dists[i].sample(shape) for i in range(len(self.dists))]
        )
        sample_from = np.random.choice(self.choose_from, size=shape[0], p=self.probas)
        return samples[np.arange(shape[0]), sample_from]


def _objective(qs: dict, sample_count: int, n_dists, trial: optuna.Trial):
    sampler_params = {"n_dists": n_dists}
    dist_probas = np.array(
        [
            trial.suggest_float(f"dist_{dist_idx}_proba", 0.0, 100.0)
            for dist_idx in range(n_dists)
        ]
    )
    dist_probas = dist_probas / dist_probas.sum()
    sampler_params = sampler_params | {
        f"dist_{dist_idx}_proba": proba for dist_idx, proba in enumerate(dist_probas)
    }

    for idx in range(n_dists):
        dist_type = trial.suggest_categorical(
            f"dist_{idx}_type", ["Normal", "Beta", "Chisquare", "Exponential"]
        )
        if dist_type in ("Normal",):
            dist_params = {
                f"dist_{idx}_loc": trial.suggest_float(f"dist_{idx}_loc", -3.0, 3.0),
                f"dist_{idx}_scale": trial.suggest_float(
                    f"dist_{idx}_scale", 0.00001, 3.0
                ),
            }
        elif dist_type in ("Beta",):
            dist_params = {
                f"dist_{idx}_alpha": trial.suggest_float(
                    f"dist_{idx}_alpha", 0.00001, 3.0
                ),
                f"dist_{idx}_beta": trial.suggest_float(
                    f"dist_{idx}_beta", 0.00001, 3.0
                ),
            }
        elif dist_type in ("Chisquare",):
            dist_params = {
                f"dist_{idx}_df": trial.suggest_float(f"dist_{idx}_df", 0.001, 15.0)
            }
        elif dist_type in ("Exponential",):
            dist_params = {
                f"dist_{idx}_lambda": trial.suggest_float(
                    f"dist_{idx}_lambda", 0.01, 2.5
                )
            }
        else:
            raise KeyError(dist_type)

        dist_params[f"dist_{idx}_type"] = dist_type

        sampler_params = sampler_params | dist_params

    sampler = MultiDistSampler(sampler_params)

    samples = sampler.sample([sample_count])
    loss = sum(abs(np.quantile(samples, q) - q_val) for q, q_val in qs.items())

    return loss

class Discriminator(torch.nn.Module):
    """
    Класс генератора-дискриминатора
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1+in_features)
        )
    
    def forward(self, x):
        out = self.model(x)
        fake_cls = torch.nn.functional.sigmoid(out[:, [-1]])
        closest_sample = out[:, :-1]
        return fake_cls, closest_sample


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
        train_epochs: Optional[int] = 1000
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

        # инициализируем сэмплеры для каждой колонки
        self.samplers = {}
        for cat_col_idx in self.categorical_indecies:
            unique_values, counts = np.unique(table[:, cat_col_idx], return_counts=True)
            counts = counts / counts.sum()  # превращаем счет в вероятность
            # добавляем сэмлер категориального распределения
            self.samplers[cat_col_idx] = MultiDistSampler(
                {
                    "n_dists": 1,
                    "dist_0_type": "Categorical",
                    "dist_0_choices": unique_values,
                    "dist_0_probas": counts,
                    "dist_0_proba": 1.0
                }
            )
        
        for num_col_idx in tqdm(
            self.numeric_indecies, desc="Инициализирую сэмлеры для числовых признаков"
        ):
            qs = {
                q: np.quantile(table[:, num_col_idx], q)
                for q in [
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.85,
                    0.9,
                    0.95,
                ]
            }
            study = optuna.create_study(
                sampler=optuna.samplers.NSGAIIISampler(population_size=30)
            )
            study.optimize(
                lambda trial: _objective(
                    qs,
                    self.sample_count_during_pick,
                    self.n_distributions_per_column,
                    trial,
                ),
                n_trials=self.n_optuna_trials_per_column,
                timeout=self.optuna_timeout_seconds
            )

            # print(study.best_value)
            # print(study.best_params)

            sampler_params = study.best_params.copy()
            sampler_params["n_dists"] = self.n_distributions_per_column
            dist_probas = np.array(
                [
                    sampler_params[f"dist_{dist_idx}_proba"]
                    for dist_idx in range(sampler_params["n_dists"])
                ]
            )
            dist_probas = dist_probas / dist_probas.sum()
            sampler_params = sampler_params | {
                f"dist_{dist_idx}_proba": proba
                for dist_idx, proba in enumerate(dist_probas)
            }

            sampler = MultiDistSampler(sampler_params)
            self.samplers[num_col_idx] = sampler

            plt.close("all")
            fig = plt.figure()
            fig.gca().hist(
                sampler.sample([self.sample_count_during_pick]),
                bins=40,
                alpha=0.5,
                label="simulated",
                density=True,
            )
            fig.gca().hist(
                table[:, num_col_idx], bins=40, alpha=0.5, label="real", density=True
            )
            fig.legend()
            col = self.column_names[num_col_idx]
            fig.suptitle(f"Аппроксимация распределения колонки {col}")
            fig.savefig(self.plot_report_dir / f"{col}.png")

        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        if self.categorical_indecies:
            self.one_hot_encoder.fit(table[:, self.categorical_indecies])

        if self.categorical_indecies and self.numeric_indecies:
            temporary_repr = np.column_stack([
                table[:, self.numeric_indecies],
                self.one_hot_encoder.transform(table[:, self.categorical_indecies])
            ])
        elif self.numeric_indecies:
            temporary_repr = table
        elif self.categorical_indecies:
            temporary_repr = self.one_hot_encoder.transform(table)
        else:
            raise ValueError("Both numeric indecies and categorical indecies are empty lists!")
        
        print(f"{temporary_repr.shape=}")

        temporary_repr = torch.as_tensor(temporary_repr).float().to(device_str)
        real_label = torch.ones((temporary_repr.shape[0], 1)).to(device_str)
        fake_label = torch.zeros((temporary_repr.shape[0], 1)).to(device_str)

        self.discriminator_generator = Discriminator(in_features=temporary_repr.shape[1]).to(device_str)
        self.optimizer = torch.optim.AdamW(self.discriminator_generator.parameters(), lr=1e-4)
        pdist = torch.nn.PairwiseDistance()
        
        print(f"Count of model params: {sum(param.numel() for param in self.discriminator_generator.parameters())}")

        print_loss = False
        train_range = tqdm(range(self.train_epochs), total=self.train_epochs, desc="Обучаю дискриминатор-генератор")
        for epoch in train_range:
            def closure():
                self.optimizer.zero_grad()
                samples = self._raw_sample(temporary_repr.shape[0])
                if self.categorical_indecies and self.numeric_indecies:
                    samples = np.column_stack([
                        samples[:, self.numeric_indecies],
                        self.one_hot_encoder.transform(samples[:, self.categorical_indecies])
                    ])
                elif self.categorical_indecies:
                    samples = self.one_hot_encoder.transform(samples)
                
                samples = torch.as_tensor(samples).float().to(device_str)

                # closest_samples - вектор сгенерированных строк таблиц
                is_real, closest_samples = self.discriminator_generator(temporary_repr)
                cls_loss_real = torch.nn.functional.binary_cross_entropy(is_real, real_label)
                reconstruct_loss_real = ((closest_samples - temporary_repr)**2).mean()
                
                is_real_samples, closest_samples_samples = self.discriminator_generator(samples)
                cls_loss_fake = torch.nn.functional.binary_cross_entropy(is_real_samples, fake_label)
                shuf = torch.randperm(temporary_repr.shape[0])
                reconstruct_loss_samples = ((closest_samples_samples - temporary_repr[shuf, :])**2).mean()
                # reconstruct_loss_samples = 0.0
                # for i in range(closest_samples_samples.shape[0]):
                #     dist_to_train = pdist(temporary_repr, closest_samples_samples[i])
                #     argmin_dist = torch.argmin(dist_to_train)
                #     closest_found_sample = temporary_repr[argmin_dist, :]
                #     reconstruct_loss_samples = reconstruct_loss_samples + ((closest_samples_samples[i, :] - closest_found_sample)**2).mean()
                # reconstruct_loss_samples = reconstruct_loss_samples / closest_samples_samples.shape[0]

                loss = cls_loss_real + cls_loss_fake + reconstruct_loss_real + reconstruct_loss_samples
                if print_loss:
                    train_range.set_postfix({
                        "epoch": epoch,
                        "cls_loss_fake": round(cls_loss_fake.item(), 4),
                        "cls_loss_real": round(cls_loss_real.item(), 4),
                        "reconstruct_loss_real": round(reconstruct_loss_real.item(), 4),
                        "reconstruct_loss_samples": round(reconstruct_loss_samples.item(), 4),
                        "loss": round(loss.item(), 4)
                    })
                
                if loss.requires_grad:
                    loss.backward()
                return loss
            self.optimizer.step(closure)
            with torch.no_grad():
                print_loss = True
                closure()
                print_loss = False
                
    def _raw_sample(self, n):
        return np.column_stack([self.samplers[i].sample([n]) for i in range(self.total_in_features)])
    
    def generate(self, n, pick_real: Optional[bool] = True, pick_closest: Optional[bool] = False):
        assert n > 1, "n should be larger than 1!"
        # n - сырая генерация. Итоговый набор может быть и пустым
        self.discriminator_generator.to("cpu").eval()
        
        samples = self._raw_sample(n)
        if self.categorical_indecies and self.numeric_indecies:
            samples = np.column_stack([
                samples[:, self.numeric_indecies],
                self.one_hot_encoder.transform(samples[:, self.categorical_indecies])
            ])
        elif self.categorical_indecies:
            samples = self.one_hot_encoder.transform(samples)
        
        samples = torch.as_tensor(samples).float()
        with torch.no_grad():
            is_real, closest = self.discriminator_generator(samples)
        
        is_real = is_real.detach().cpu().numpy()
        closest = closest.detach().cpu().numpy()
        
        if (pick_real and pick_closest) or pick_closest:
            to_return = closest
        elif pick_real:
            print(f"{is_real.squeeze().shape=}")
            to_return = closest[is_real.squeeze() > 0.5, :]
        else:
            raise ValueError("At least one of pick_real and pick_closest should be True!")
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
