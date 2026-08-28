from typing import List, Optional

from torchdata.stateful_dataloader import StatefulDataLoader

from boa.data.of_batch import OFCollater


class OFDataLoader(StatefulDataLoader):
    r"""Data loader for :class:`OFData` samples.

    ``OFData`` cannot be batched by a generic collater -- scdp's raises
    ``TypeError: DataLoader found invalid type`` on it -- and the difference is
    more than cosmetic: ``message_passing_matrix`` is an
    :class:`~boa.data.overlap_matrix.OverlapMatrix`, which
    :class:`~boa.data.of_batch.OFCollater` zero-pads and stacks to
    ``(n_mol, n_ao, n_ao)`` rather than concatenating along dim 0. Batched any
    other way, per-molecule quantities such as ``tr(D S)`` silently stop being
    computable.

    Args:
        dataset: The dataset from which to load the data.
        batch_size: How many samples per batch to load. (default: :obj:`1`)
        shuffle: If set to :obj:`True`, the data will be reshuffled at every
            epoch. (default: :obj:`False`)
        follow_batch: Creates assignment batch vectors for each key in the list.
        exclude_keys: Will exclude each key in the list.
        list_keys: Collated into a plain list rather than a tensor, for
            per-sample matrices whose size varies between molecules.
        **kwargs: Additional arguments of :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        list_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.list_keys = list_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            # OFCollater takes `dataset` first and never uses it; pass the rest
            # by keyword so nothing shifts a slot left.
            collate_fn=OFCollater(
                None,
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
                list_keys=list_keys,
            ),
            **kwargs,
        )


class ProbeDataLoader(OFDataLoader):
    """:class:`OFDataLoader` that additionally accepts ``n_probe``.

    ``n_probe`` plays no part in batching. The probe count is applied by the
    dataset (``SmallDensityDataset.subsample_grid``) and by the ``SampleProbe``
    transform, both long before collation; the loader and the collater only ever
    stored it. It is accepted here so the grid configs, which pass it through
    :class:`~boa.data.datamodule.ProbeDataModule`, keep working unchanged.

    Anything that does not sample a grid should use :class:`OFDataLoader`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        n_probe: int = 200,
        **kwargs,
    ):
        self.n_probe = n_probe
        super().__init__(dataset, batch_size, shuffle, **kwargs)
