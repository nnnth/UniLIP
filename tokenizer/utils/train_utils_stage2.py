"""Training utils for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import open_clip
import sys 
sys.path.append('.')
from data import TextImageDataset
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel
from modeling.modules import loss_map
from modeling import model_map
from evaluator import Evaluator
from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation, make_viz_from_samples_t2i_generation
from torchinfo import summary


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok"):
    """Creates model and loss module."""
    logger.info("Creating model and loss module.")
    model_cls = model_map[model_type]
    loss_cls = loss_map[config.losses.name]
     
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config) if loss_cls is not None else None

    # Print Model for sanity check.
    if accelerator.is_main_process:
        # if model_type in ["titok","vae","ae_vit","ae_vit_pretrain","ae_vit_pretrain_add", "ae_vit_pretrain_distill", "ae_vit_pretrain_probe", "ae_vit_pretrain_probe_v2"]:
        input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
        model_summary_str = summary(model, input_size=input_size, depth=5,
        col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
        logger.info(model_summary_str)

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="titok", need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude_encoder = (lambda n, p: ('encoder' in n or 'mlp1' in n) and (p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n))
    include_encoder = lambda n, p: not exclude_encoder(n, p) and ('encoder' in n or 'mlp1' in n) 
    exclude_other = (lambda n, p: ('encoder' not in n and 'mlp1' not in n) and (p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n))
    include_other = lambda n, p: not exclude_other(n, p) and ('encoder' not in n and 'mlp1' not in n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params_encoder = [p for n, p in named_parameters if exclude_encoder(n, p) and p.requires_grad]
    rest_params_encoder = [p for n, p in named_parameters if include_encoder(n, p) and p.requires_grad]
    gain_or_bias_params_other = [p for n, p in named_parameters if exclude_other(n, p) and p.requires_grad]
    rest_params_other = [p for n, p in named_parameters if include_other(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params_encoder, "weight_decay": 0., 'lr':learning_rate*0.1},
            {"params": rest_params_encoder, "weight_decay": optimizer_config.weight_decay, 'lr':learning_rate*0.1},
            {"params": gain_or_bias_params_other, "weight_decay": 0., 'lr':learning_rate},
            {"params": rest_params_other, "weight_decay": optimizer_config.weight_decay, 'lr':learning_rate},
        ],
        # lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    # if (config.model.vq_model.finetune_decoder or model_type in ["tatitok",'vae','ae_vit','ae_vit_pretrain','ae_vit_pretrain_add', 'ae_vit_pretrain_distill', 'ae_vit_pretrain_probe', 'ae_vit_pretrain_probe_v2']) and need_discrminator:
    if need_discrminator:
        exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
        include = lambda n, p: not exclude(n, p)
        print("set discriminator")
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        print("no discriminator")
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for TiTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

     
    print("max train samples", config.experiment.max_train_examples)
    print("change max train samples if use different datasets!!!!!!!")
    dataset = TextImageDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
        dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
        dataset_with_text_label=dataset_config.get("dataset_with_text_label", False),
        res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
        normalize_mean=preproc_config.get('normalize_mean', [0., 0., 0.]),
        normalize_std=preproc_config.get('normalize_std', [1., 1., 1.]),
        sample_ratio=dataset_config.get('sample_ratio', [])
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    evaluator = Evaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=False,
        enable_codebook_entropy_measure=False,
    )
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    model_type="titok"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model, loss_module]):
            if model_type == "titok":
                reconstructed_images, extra_results_dict = model(images)
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )
            else:
                raise NotImplementedError

            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                if 'train/distill_loss' in autoencoder_logs:
                    distill_loss = autoencoder_logs['train/distill_loss']
                else:
                    distill_loss = 0.0
                if accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                    if 'train/logits_real' in discriminator_logs:
                        logits_real = discriminator_logs['train/logits_real']
                    else:
                        logits_real = 0.0
                    if 'train/logits_fake' in discriminator_logs:
                        logits_fake = discriminator_logs['train/logits_fake']
                    else:
                        logits_fake = 0.0
                else:
                    logits_fake = logits_real = 0.0
                                    
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                    f"Distill Loss: {distill_loss:0.4f} "
                    f"Logits Real: {logits_real:0.4f} "
                    f"Logits Fake: {logits_fake:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_images(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        model_type=model_type,
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step

@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    model_type="titok",
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )

        original_images = torch.clone(images)
        if model_type == "titok":
            reconstructed_images, model_dict = local_model(images)
        else:
            raise NotImplementedError

        reconstructed_images = torch.clamp(reconstructed_images, -1.0, 1.0)
        # Quantize to uint8
        reconstructed_images = (torch.round((reconstructed_images+1)/2 * 255.0) - 127.5) / 127.5
        evaluator.update(original_images, reconstructed_images.squeeze(2), None)
            
    model.train()
    return evaluator.result()


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="titok"):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
    
    if model_type == "titok":
        reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)
    if config.dataset.preprocessing.get('imagenet_norm', False):
        mean, std = config.dataset.preprocessing.normalize_mean, config.dataset.preprocessing.normalize_std
        std = torch.tensor(std).to(original_images.device)
        mean = torch.tensor(mean).to(original_images.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        # inputs is normalized by imagenet, while reconstruction predict [-1,1]
        # here align both to [0, 1]
        original_images = original_images * std + mean
        reconstructed_images = (reconstructed_images + 1) / 2
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        # NOTE: for name like 0335161/0000001.png, contain folder
        fnames[i] = fnames[i].split('/')[-1]
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()
def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)