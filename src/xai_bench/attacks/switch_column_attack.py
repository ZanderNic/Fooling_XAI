import numpy as np
import pandas as pd
from typing import Optional, Literal, Union, Any, overload
from itertools import permutations
import math

from xai_bench.base import BaseModel, BaseDataset, BaseAttack
from tqdm.rich import tqdm, trange

class ColumnSwitchAttack(BaseAttack):
    """
    model: the model to use for fitting (finding best switches)
    task: classification or regression, passed on to model.predict
    """
    def __init__(self, model:BaseModel, task: Literal["classification","regression"]):
        super().__init__(model, task=task)
        self.top_combi: Optional[list] = None
    
    """
    will create new adv data by switching columns according to given combi of indices
    """
    @overload
    def _switch_columns(self, X_train:pd.DataFrame, combi:np.ndarray) -> pd.DataFrame:
        pass
    @overload
    def _switch_columns(self, X_train:np.ndarray, combi:np.ndarray) -> np.ndarray:
        pass
    def _switch_columns(self, X_train, combi:np.ndarray):
        x_adv = X_train.copy()
        if isinstance(x_adv,pd.DataFrame):
            # is whoel dataframe
            tmp: pd.Series = x_adv.iloc[:,combi[0]] # type: ignore # save first/previous
            # iterate over every entry, but start at second
            for i in range(1,len(combi)):
                tmp2 = x_adv.iloc[:,combi[i]] # write current to tmp2
                x_adv.iloc[:,combi[i]] = tmp # overwrtie current with previous
                tmp = tmp2 # wrtie current to tmp (will be previous to next)
            # wrap last element around
            x_adv.iloc[:,combi[0]] = tmp
        else:
            # is only single 1d array
            tmp: Any = x_adv[combi[0]] # save first/previous
            # iterate over every entry, but start at second
            for i in range(1,len(combi)):
                tmp2 = x_adv[combi[i]] # write current to tmp2
                x_adv[combi[i]] = tmp # overwrtie current with previous
                tmp = tmp2 # wrtie current to tmp (will be previous to next)
            # wrap last element around
            x_adv[combi[0]] = tmp
        # switch columns
        return x_adv
    
    """
    Will switch given combi columns and then let model predict (needs already to be trained)
    """
    def _evaluate(self,X_train:pd.DataFrame, combi:np.ndarray) -> float:
        x_adv = self._switch_columns(X_train,combi)
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
    def fit(self, dataset:BaseDataset, n_switches:int, max_tries:Optional[int]=None):
        assert dataset.X_train is not None and dataset.y_train is not None, "Dataset needs to be loaded"
        assert n_switches>=2, "One is not an option"
        if dataset.numerical_features is None:
            raise ValueError(f"This dataset ({dataset}) has the numerical_features attribute not set, which is needed for a ColumnSwitchAttack!")
        assert dataset.features, "Dataset needs features"
        feature_indexes = [dataset.features.feature_names_model.index(f) for f in dataset.numerical_features] # TODO: get from feature mapping

        top_score = np.inf
        top_combi = []  # combi means lsit in index with data switching to the right. E.g. [1,4,6] would result in 1->4->6->1, so column 1 now has data from column 6, column 4 has now date from column 1 and column 6 has now data from column 4.
                        # So a combi of just two columns will jsut switch (and [1,0] is the same as [0,1]. This does not apply to n>2)
        tbar = tqdm(unit="tries")
        tbar.postfix = "Top Combi: []"
        if max_tries is not None:
            tbar.total = max_tries
            tbar.desc = "[CSA] Fitting MAX"
            for _ in tqdm(range(max_tries)):
                combi = np.random.choice(feature_indexes,size=(n_switches,),replace=False)
                score = self._evaluate(dataset.X_train,combi)
                if score<= top_score:
                    top_score = score
                    top_combi = list(combi)
                    tbar.postfix = f"Top Combi: {top_combi}"
                tbar.update()
        
        else:
            tbar.total = math.factorial(n_switches)
            tbar.desc = "[CSA] Fitting ALL"
            combinations = permutations(feature_indexes,n_switches)
            for combi in combinations:
                # count down max tries if wanted
                score = self._evaluate(dataset.X_train,np.asarray(combi))
                if score<= top_score:
                    top_score = score
                    top_combi = list(combi)
                    tbar.postfix = f"Current best Combi: {top_combi}"
                tbar.update()
        tbar.close()
        self.top_combi = top_combi
        return top_combi

    def _generate(self, x: np.ndarray) -> np.ndarray:
        if self.top_combi is None:
            raise RuntimeError("Fit the attack first before generating adverserial data.")
        return self._switch_columns(x,self.top_combi) # type: ignore