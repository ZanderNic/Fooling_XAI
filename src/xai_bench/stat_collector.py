from typing import Optional

class StatCollector():
    def __init__(self, obj, comment:Optional[str]=None):
        self.calls_samples = 0
        self.calls = 0
        self.obj = obj
        self.comment =comment

    def __call__(self, num_samples:int = 1) -> None:
        self.calls+=1
        self.calls_samples += num_samples

    def __str__(self):
        return f"StatsCollector(obj={self.obj}, comment={self.comment}, calls={self.calls}, calls_samples={self.calls_samples})"
