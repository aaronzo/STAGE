
from typing import *
from torch_geometric.loader.dataloader import Collater as PygCollater

class Collater(PygCollater):
    def __init__(
        self,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            None,  # not actually used
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )
    
    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, Mapping):
            mapping = {
                key: [data[key] for data in batch]
                for key in self.exclude_keys
                if key in elem
            }
            mapping.update({
                key: self([data[key] for data in batch])
                for key in elem
                if key not in self.exclude_keys
            })
            return mapping
        return super().__call__(batch)
