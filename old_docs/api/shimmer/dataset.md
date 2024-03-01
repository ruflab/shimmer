Module shimmer.dataset
======================

Classes
-------

`RepeatedDataset(dataset: shimmer.dataset._SizedDataset, min_size: int, drop_last: bool = False)`
:   Repeats a dataset to have at least a minimum size.
    
    Params:
        dataset: dataset to repeat
        min_size (int): minimum amount of element in the final dataset
        drop_last (bool): whether to remove overflow when repeating the
            dataset.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic