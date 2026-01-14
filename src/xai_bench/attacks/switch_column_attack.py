import numpy as np
import pandas as pd
from typing import Optional, Literal, Union, Any, overload
from itertools import permutations
import math

from xai_bench.base import BaseModel, BaseDataset, BaseAttack
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from xai_bench.console import console

class ColumnSwitchAttack(BaseAttack):
    """
    model: the model to use for fitting (finding best switches)
    task: classification or regression, passed on to model.predict
    """
    def __init__(self, model:BaseModel, task: Literal["classification","regression"], epsilon:Optional[float]=None):
        super().__init__(model, task=task,epislon=epsilon,stats=[self,"Swtich attack"])
        self.top_combi: Optional[list] = None
    
    """
    will create new adv data by switching columns according to given combi of indices
    """
    @overload
    def _switch_columns(self, X_train:pd.DataFrame, combi:list[int]) -> pd.DataFrame:
        pass
    @overload
    def _switch_columns(self, X_train:np.ndarray, combi:list[int]) -> np.ndarray:
        pass
    def _switch_columns(self, X_train, combi:list[int]):
        x_adv = X_train.copy()
        if isinstance(x_adv,pd.DataFrame):
            switched_idx = self._switched(list(range(len(x_adv.columns))),combi=combi)
            for i,idx in enumerate(switched_idx):
                x_adv.iloc[:,i] = X_train.iloc[:,idx]
        elif isinstance(x_adv,np.ndarray):
            switched_idx = self._switched(list(range(x_adv.shape[0])),combi=combi)
            for i, idx in enumerate(switched_idx):
                x_adv[i] = X_train[idx]
        # switch columns
        return x_adv
    
    def _switched(self, indices:list[int], combi:list[int]) -> list[int]:
        # map each element in combi to the element that precedes it in the cyclic order
        prev = {}
        n = len(combi)
        for i, v in enumerate(combi):
            prev[v] = combi[(i - 1) % n]

        # for every index, return its replacement if it appears in combi, else itself
        return [prev.get(i, i) for i in indices]

    """
    Will switch given combi columns and then let model predict (needs already to be trained)
    """
    def _evaluate(self,X_train:pd.DataFrame, combi:np.ndarray) -> np.ndarray:
        x_adv = self._switch_columns(X_train,combi.tolist())
        l1 = self._prediction_distance(X_train,x_adv)
        del x_adv
        return l1

    """
    Will determine best (numerical) column switches bsed on n_switches and 
    model: The model to fit the best column switches on
    dataset: containg trainingdata to fit on
    n_switches: number of ccolumns to permutate
    max_tries: If not none, pick randomly from permutations until max_tries is reached (with possible repeats). Usefull if n_switches is high, as number of combinations is (n_switches)!
    """
    def fit(self, dataset:BaseDataset, n_switches:int, max_tries:Optional[int]=None, numerical_only:bool=True):
        self.stats("fit")
        assert dataset.X_train is not None and dataset.y_train is not None, "Dataset needs to be loaded"
        assert n_switches>=2, "One is not an option"
        assert n_switches<=len(dataset.X_train.columns), "Cant switch more columns than the dataset has"
        if dataset.numerical_features is None:
            raise ValueError(f"This dataset ({dataset}) has the numerical_features attribute not set, which is needed for a ColumnSwitchAttack!")
        assert dataset.features, "Dataset needs features"
        if numerical_only:
            feature_indexes = [dataset.features.feature_names_model.index(f) for f_name in dataset.numerical_features for f in dataset.feature_mapping[f_name]] # TODO: get from feature mapping
        else:
            feature_indexes = list(range(len(dataset.features.feature_names_model)))
        console.print(f"[bold #ed1cdf][CSA][/] [#f7c8f3] Fitting on features: {feature_indexes} (aka. {[dataset.features.feature_names_model[f] for f in feature_indexes]})")

        top_score = np.inf
        top_combi: list[int] = []  # combi means lsit in index with data switching to the right. E.g. [1,4,6] would result in 1->4->6->1, so column 1 now has data from column 6, column 4 has now date from column 1 and column 6 has now data from column 4.
                        # So a combi of just two columns will jsut switch (and [1,0] is the same as [0,1]. This does not apply to n>2)
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.completed}/{task.total}",
            "[progress.percentage]({task.percentage:>3.1f}%)",
            "[bold #ed1cdf]•[/]",
            TimeElapsedColumn(),
            "[bold #ed1cdf]/[/]",
            TimeRemainingColumn(),
            "[bold #ed1cdf]•[/]",
            TextColumn("[red]Top Combi: {task.fields[combi]}"),
            console=console,
            transient=True
        )
        try:
            # tbar.postfix = "Top Combi: []"
            if max_tries is not None:
                task = progress.add_task("[bold #ed1cdf][CSA][/] [#f7c8f3] Fitting MAX...",total=max_tries, combi="")
                progress.start()
                for _ in range(max_tries):
                    combi = np.random.choice(feature_indexes,size=(n_switches,),replace=False)
                    scores = self._evaluate(dataset.X_train,combi)
                    score = scores.mean()
                    if score<= top_score:
                        top_score = score
                        top_combi = [int(i) for i in combi]
                        progress.update(task,advance=None,combi=str(top_combi))
                    progress.update(task,advance=1)
            
            else:
                task = progress.add_task("[bold #ed1cdf][CSA][/] [#f7c8f3] Fitting ALL...",total=math.factorial(len(feature_indexes))/math.factorial(len(feature_indexes)-n_switches), combi="")
                progress.start()
                combinations = permutations(feature_indexes,n_switches)
                for combi in combinations:
                    # count down max tries if wanted
                    scores = self._evaluate(dataset.X_train,np.asarray(combi))
                    score = scores.mean()
                    if score<= top_score:
                        top_score = score
                        top_combi = [int(i) for i in combi]
                        progress.update(task,advance=None,combi=str(top_combi))
                    progress.update(task,advance=1)
        finally:
            progress.stop()
        self.top_combi = top_combi
        console.print(f"[bold #ed1cdf][CSA][/] [#f7c8f3] Found best combi: [#cbe9f5]{self.top_combi}[/] (meaning [#cbe9f5]{feature_indexes}[/] -> [#cbe9f5]{self._switched(feature_indexes,self.top_combi)}[/] / [#cbe9f5]{[dataset.features.feature_names_model[f] for f in feature_indexes]}[/] -> [#cbe9f5]{[dataset.features.feature_names_model[f] for f in  self._switched(feature_indexes,self.top_combi)]}[/])")
        return top_combi

    def _generate(self, x: np.ndarray) -> np.ndarray:
        if self.top_combi is None:
            raise RuntimeError("Fit the attack first before generating adverserial data.")
        switched = self._switch_columns(x,self.top_combi)
        valid, _ = self.is_attack_valid(np.expand_dims(x,axis=0),np.expand_dims(switched,axis=0),self.epsilon)
        return  switched if valid else x