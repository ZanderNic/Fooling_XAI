from typing import Optional, overload, Union
from numpy import ndarray
from pandas import DataFrame



class StatCollector():
    def __init__(self, obj, comment:Optional[str]=None):
        self.calls_samples = {}
        self.calls = {}
        self.obj = obj
        self.comment =comment
        self.active = True

    @overload
    def __call__(self, func:str, num_samples:Optional[int]=None) -> None:
        pass
    @overload
    def __call__(self, func:str, num_samples:Optional[ndarray]=None) -> None:
        pass
    @overload
    def __call__(self, func:str, num_samples:Optional[DataFrame]=None) -> None:
        pass
    def __call__(self, func, num_samples = None):
        if not self.active:
            return
        plus = 0
        if num_samples is None:
            plus = 1
        elif isinstance(num_samples,DataFrame):
            plus = len(num_samples)
        elif isinstance(num_samples,ndarray):
            plus = num_samples.shape[0] if num_samples.ndim>=2 else 1
        elif isinstance(num_samples,int):
            plus =num_samples
        else:
            raise ValueError(f"num samples is neither dataframe, ndarray, int or None, something is wrong :( [{num_samples},{type(num_samples)}]")

        if self.calls.get(func) is None:
            self.calls[func] = 0
        if self.calls_samples.get(func) is None:
            self.calls_samples[func] = 0
        self.calls[func]+=1
        self.calls_samples[func] += plus

    def __str__(self):
        return f"StatsCollector(obj={self.obj}, comment={self.comment}, calls={self.calls}, calls_samples={self.calls_samples})"
    

    """
    Called to get all stats of all objects passed to this function
    """
    @staticmethod
    def collect(*args)->list[Union[str,dict]]:
        out = ["",{}]
        for arg in args:
            out[0] +="\n"+arg.stats.__str__()
            out[1][arg.__class__.__name__] = {"calls":arg.stats.calls,"calls_samples":arg.stats.calls_samples}
        return out
    
    """
    Call to activate counting again. Counting is per default active
    """
    def activate(self):
        self.active = True

    """
    Call to deactivate stats collection. Usefull for e.g. model training
    """
    def deactiavte(self):
        self.active = False