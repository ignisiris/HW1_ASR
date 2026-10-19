from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.batch_sampler import LengthBucketBatchSampler
from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


class TrainDataLoader(DataLoader):
    epoch = 0

    def __iter__(self):
        self.batch_sampler.set_epoch(self.epoch)
        self.epoch += 1
        return super().__iter__()


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partition init
        dataset = instantiate(
            config.datasets[dataset_partition], text_encoder=text_encoder
        )  # instance transforms are defined inside

        batch_size = min(config.dataloader.batch_size, len(dataset))
        if dataset_partition == "train":
            lengths = [item["audio_len"] for item in dataset._index]
            batch_sampler = LengthBucketBatchSampler(
                lengths, batch_size, seed=config.trainer.get("seed", 42), drop_last=True
            )
            loader_config = dict(config.dataloader)
            loader_config.pop("batch_size")
            loader_config["_target_"] = "src.datasets.data_utils.TrainDataLoader"
            partition_dataloader = instantiate(
                loader_config,
                dataset=dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                worker_init_fn=set_worker_seed,
            )
        else:
            partition_dataloader = instantiate(
                config.dataloader,
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=False,
                worker_init_fn=set_worker_seed,
            )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
