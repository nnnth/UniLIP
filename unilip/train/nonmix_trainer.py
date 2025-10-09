import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from transformers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
from typing import Iterator, Optional, List

class DistributedTaskTypeBatchSampler(BatchSampler):
    """
    一个支持分布式训练的BatchSampler，它首先按任务类型对数据进行分组，
    然后在每个epoch中，保证每个副本（GPU）拿到不重复的、随机打乱的批次。

    核心逻辑：
    1. 在所有副本上生成一个完全相同的、全局的批次列表（batches）。
    2. 根据副本数量（world_size）对这个全局批次列表进行填充或截断，使其长度能被整除。
    3. 每个副本（rank）根据自己的排名，从全局批次列表中切分出自己需要处理的部分。
    4. `set_epoch()` 方法确保每个epoch的随机种子都不同，从而实现不同的数据打乱顺序。
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 drop_last: bool = False):
        """
        Args:
            dataset (Dataset): 需要采样的数据集。
            batch_size (int): 每个批次的大小。
            shuffle (bool): 是否在每个 epoch 开始时打乱数据。
            num_replicas (int, optional): 分布式训练中的进程数。如果为None，则从 dist.get_world_size() 获取。
            rank (int, optional): 当前进程的排名。如果为None，则从 dist.get_rank() 获取。
            drop_last (bool): 如果为 True，则丢弃最后一个不完整的批次。在这里，它意味着丢弃无法被副本数整除的尾部批次。
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        # 与原始 TaskTypeBatchSampler 相同的逻辑，构建 type 到 indices 的映射
        type_list = dataset.list_data_dict["type"]
        self.type_to_indices = {}
        for idx, t in enumerate(type_list):
            self.type_to_indices.setdefault(t, []).append(idx)
        
        # 计算所有可能的完整批次
        self.total_batches = self._calculate_total_batches()
        
        # 计算每个副本的样本数（这里是批次数）
        if self.drop_last and self.total_batches % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            self.num_samples = math.ceil((self.total_batches - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.total_batches / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas

    def _calculate_total_batches(self) -> int:
        # 计算在 drop_last=True 的情况下，整个数据集可以产生多少个完整的批次
        return sum(len(indices) // self.batch_size for indices in self.type_to_indices.values())

    def __iter__(self) -> Iterator[List[int]]:
        # 1. 生成全局的批次列表 (在所有 rank 上都相同)
        g = torch.Generator()
        g.manual_seed(self.epoch)  # 使用 epoch 作为种子，确保每个 epoch 的 shuffle 不同但所有 rank 相同

        all_batches = []
        for t, indices in self.type_to_indices.items():
            # 为每个类型的索引列表创建副本进行操作
            idxs = list(indices)
            if self.shuffle:
                # 使用带种子的生成器进行打乱
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            
            # 按 batch_size 切分，并只保留完整的批次
            for i in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                all_batches.append(idxs[i : i + self.batch_size])

        # 如果需要，再次打乱所有批次的顺序
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]
        
        # 2. 填充或截断批次列表以适应分布式设置
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(all_batches)
            if padding_size > 0:
                all_batches += all_batches[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            all_batches = all_batches[:self.total_size]

        assert len(all_batches) == self.total_size

        # 3. 为当前 rank 切分出子集
        # subsample
        indices_on_this_rank = all_batches[self.rank : self.total_size : self.num_replicas]
        assert len(indices_on_this_rank) == self.num_samples

        return iter(indices_on_this_rank)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class NonMixTrainer(Trainer):
    def get_train_dataloader(self):
        """
        重写此方法以使用我们自定义的分布式批次采样器。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 使用新的分布式采样器
        batch_sampler = DistributedTaskTypeBatchSampler(
            self.train_dataset,
            batch_size=self._train_batch_size, # 注意：使用 _train_batch_size (per_device)
            shuffle=True,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            # Dataloader需要 set_epoch 方法，通过将其设置为 True 来自动调用
            # 但是，Hugging Face Trainer 会手动调用，所以这里可以不设置
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        这个方法现在可以被 get_train_dataloader 覆盖，但为了保持完整性，
        我们可以让它返回 None，因为我们使用的是 batch_sampler。
        或者，如果你的逻辑在某些情况下回退到使用这个方法，
        你需要确保它不会与 get_train_dataloader 中的 batch_sampler 冲突。
        
        在当前实现中，由于我们重写了 get_train_dataloader，这个方法不会被调用来创建训练数据加载器。
        """
        # 返回 None，因为我们使用的是 batch_sampler
        return None

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

