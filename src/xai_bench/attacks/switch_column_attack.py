import numpy as np
import pandas as pd
from typing import Optional, Literal, overload
import math

from xai_bench.base import BaseModel, BaseDataset, BaseAttack, BaseMetric,BaseExplainer
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from xai_bench.console import console

class ColumnSwitchAttack(BaseAttack):
    def __init__(
        self,
        model: BaseModel,
        task: Literal["classification", "regression"],
        dataset: BaseDataset,
        metric:BaseMetric,
        explainer: BaseExplainer,
        epsilon: Optional[float] = None,
        n_switches: int = 3,
        max_tries: int = 1000,
        numerical_only: bool = True,
        explain_ratio:float=0.1
    ):
        super().__init__(
            model,
            task=task,
            epsilon=epsilon,
            stats=[self, "Swtich attack"],
            dataset=dataset,
        )

        assert dataset.X_train is not None and dataset.y_train is not None, (
            "Dataset needs to be loaded"
        )
        assert n_switches >= 2, "One is not an option"
        assert n_switches <= len(dataset.X_train.columns), (
            "Cant switch more columns than the dataset has"
        )
        assert dataset.numerical_features is not None, f"This dataset ({dataset}) has the numerical_features attribute not set, which is needed for a ColumnSwitchAttack!"
        assert dataset.features, "Dataset needs features"

        self.max_tries = max_tries
        self.n_switches = n_switches
        self.numerical_only = numerical_only
        self.metric = metric
        self.explainer = explainer
        self.explain_ratio = explain_ratio # how much to decrese explain samples for faster results

        if numerical_only:
            self.feature_indexes = [
                dataset.features.feature_names_model.index(f)
                for f_name in dataset.numerical_features
                for f in dataset.feature_mapping[f_name]
            ]
        else:
            self.feature_indexes = list(range(len(dataset.features.feature_names_model)))

        console.print(
            f"[bold #ed1cdf][CSA][/][#f7c8f3] Initialising on features: {self.feature_indexes} (aka. {[dataset.features.feature_names_model[f] for f in self.feature_indexes]})"
        )

    def fit(self):
        pass

    def _set_max_tries(self):
        if self.max_tries is None:
            self.max_tries = int(math.factorial(len(self.feature_indexes)) / math.factorial(len(self.feature_indexes) - self.n_switches))

    # nd array of dim (feature,)
    def _generate(self, x:np.ndarray) -> np.ndarray:
        with self._get_progress_bar() as progress:
            # if no given take all
            self._set_max_tries()
            task = progress.add_task(
                        "[bold #ed1cdf][CSA][/] [#f7c8f3] Generating One Sample...",
                        total=self.max_tries,
                        combi="",
                    )
            
            best_attack = {"score":0, "combi":None}
            for _ in range(self.max_tries):
                combi = np.random.choice(
                            self.feature_indexes, size=(self.n_switches,), replace=False
                        ).tolist()
                x_tmp = self._switch_columns(x, combi)
                b, _ = self.is_attack_valid(np.expand_dims(x,axis=0),np.expand_dims(x_tmp,axis=0))
                if not b:
                    # attack is not valid, so dont even evaluate metrics
                    progress.update(task, advance=1)
                    continue
                # get explainations:
                x_exp, x_tmp_exp = self.explainer.explain(np.expand_dims(x,axis=0),int(self.explainer.num_samples/10)), self.explainer.explain(np.expand_dims(x_tmp,axis=0),int(self.explainer.num_samples/10))
                # evalute based on given emtric
                score = self.metric.compute(x_exp,x_tmp_exp)
                if score>best_attack["score"]:
                    best_attack["combi"] = combi
                    best_attack["score"] = score
                    progress.update(task, advance=1,combi=combi)
                    continue
                progress.update(task, advance=1)
        # return attack
        if best_attack["combi"] is None:
            # no valid attack was found
            console.print("[bold #ed1cdf][CSA][/] [red] Could not find valid attack for sample.")
            return x
        else:
            return self._switch_columns(x,best_attack["combi"])
        
    # arrayo f dim (n,feature)
    def _generate_multiple(self, X:np.ndarray)->np.ndarray:
        with self._get_progress_bar(single=False) as progress:
            self._set_max_tries()
            task = progress.add_task(
                        f"[bold #ed1cdf][CSA][/] [#f7c8f3] Generating All {len(X)} Samples...",
                        total=self.max_tries,
                        combi="",
                    )
            best_scores = np.zeros((len(X)))
            best_combis = [None for _ in range(len(X))]
            for _ in range(self.max_tries):
                combis = np.stack([np.random.choice(
                            self.feature_indexes, size=(self.n_switches,), replace=False
                        ) for _ in range(len(X))])
                X_tmp = np.stack([self._switch_columns(x, combi.tolist()) for x,combi in zip(X,combis)])
                Bs, _ = self.is_attack_valid(X,X_tmp)
                X_exp, X_tmp_exp = self.explainer.explain(X,int(self.explainer.num_samples/10)), self.explainer.explain(X_tmp,int(self.explainer.num_samples/10))
                scores = self.metric.compute(X_exp,X_tmp_exp)
                save_mask = Bs & (scores>best_scores)
                best_combis = [combi if b else best for b, combi, best in zip(save_mask,combis,best_combis)]
                progress.update(task, advance=1,combis=int(save_mask.sum()))

        return np.stack([self._switch_columns(x, combi) if combi is not None else x for x,combi in zip(X,best_combis)])



    def generate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self.stats("generate", x)
        if x.ndim == 2:
            return self._generate_multiple(x)
        else:
            return self._generate(x)

    def _get_progress_bar(self,single:bool=True):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.completed}/{task.total}",
            "[progress.percentage]({task.percentage:>3.1f}%)",
            "[bold #ed1cdf]•[/]",
            TimeElapsedColumn(),
            "[bold #ed1cdf]/[/]",
            TimeRemainingColumn(),
            "[bold #ed1cdf]•[/]",
            TextColumn("[red]Top Combi: {task.fields[combi]}") if single else TextColumn("[red]# of better combis: {task.fields[combi]}"),
            console=console,
            transient=True,
        )

    """
    will create new adv data by switching columns according to given combi of indices
    """

    @overload
    def _switch_columns(self, X_train: pd.DataFrame, combi: list[int]) -> pd.DataFrame:
        pass

    @overload
    def _switch_columns(self, X_train: np.ndarray, combi: list[int]) -> np.ndarray:
        pass

    def _switch_columns(self, X_train, combi: list[int]):
        x_adv = X_train.copy()
        if isinstance(x_adv, pd.DataFrame):
            switched_idx = self._switched(list(range(len(x_adv.columns))), combi=combi)
            for i, idx in enumerate(switched_idx):
                x_adv.iloc[:, i] = X_train.iloc[:, idx]
        elif isinstance(x_adv, np.ndarray):
            switched_idx = self._switched(list(range(x_adv.shape[0])), combi=combi)
            for i, idx in enumerate(switched_idx):
                x_adv[i] = X_train[idx]
        # switch columns
        return x_adv

    def _switched(self, indices: list[int], combi: list[int]) -> list[int]:
        # map each element in combi to the element that precedes it in the cyclic order
        prev = {}
        n = len(combi)
        for i, v in enumerate(combi):
            prev[v] = combi[(i - 1) % n]

        # for every index, return its replacement if it appears in combi, else itself
        return [prev.get(i, i) for i in indices]