"""Training script for TiTok.

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

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
import sys 
sys.path.append('.')
from utils.logger import setup_logger
import pprint

from utils.train_utils_stage2 import (
    get_config, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    eval_reconstruction)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="UniLIP", log_level="INFO",
     output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        
    accelerator.wait_for_everyone()

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type=config.model.name)
    
    print(f"load {config.checkpoint_path}")
    pretrain_ckpt = torch.load(config.checkpoint_path)
    msgs = model.load_state_dict(pretrain_ckpt)
    print(msgs)

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model = accelerator.prepare(model)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    eval_scores = eval_reconstruction(
        model,
        eval_dataloader,
        accelerator,
        evaluator,
        model_type="titok",
    )
    logger.info(pprint.pformat(eval_scores))

if __name__ == "__main__":
    main()