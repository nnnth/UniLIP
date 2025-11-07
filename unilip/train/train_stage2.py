import os
import io
import copy
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
from unilip.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX
from torch.utils.data import Dataset
from unilip.train.nonmix_trainer import NonMixTrainer
from unilip import conversation as conversation_lib
from unilip.model import *
from unilip.mm_utils import tokenizer_image_token
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from unilip.conversation import Conversation, SeparatorStyle
from tqdm import tqdm
from copy import deepcopy 

ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    gen_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    n_und_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    gen_pooling: Optional[str] = field(default="all")  # options are: pool2d_3, pool2d_9, seq_3, seq_9, seq_27
    unilip_path: Optional[str] = field(default="")
    unilip_factor: Optional[float] = field(default=5.85)
    weighting_scheme: Optional[str] = field(default='logit_normal')
    fix_dit: bool = field(default=False)
    fix_connect: bool = field(default=False)
    fix_vit: bool = field(default=True)
    fix_llm: bool = field(default=True)
    connect_layer: int=field(default=6)
    mllm_path: Optional[str] = field(default="")
    mllm_hf_path: Optional[str] = field(default="")
    vae_path: Optional[str] = field(default="")
    dit_path: Optional[str] = field(default="")



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    gen_image_folder: Optional[str] = field(default=None)
    edit_image_folder: Optional[str] = field(default=None)
    gen_repeat: Optional[int] = field(default=1)
    edit_repeat: Optional[int] = field(default=1)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    bf16: bool = True
    pretrain_path : str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=[""]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()
    

    # Only save Adapter
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # NOTE: default to 256 tokens for 448x448
    und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
    gen_placeholder = ""
    # "[IMG]" + "<image>" * data_args.n_query + "[/IMG]"
    inst_type = None
    for source in sources:  # [instance]
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
                inst_type = "gen"
    return sources, inst_type

def preprocess_internvl(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\n' }}{% elif content['type'] == 'video' %}{{ '<video>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        # input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    input_ids = input_ids[:, :-2]
    targets = targets[:, :-2]
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )




def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_IMAGE_TOKEN in source[1]['value']
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "internvl":
        return preprocess_internvl(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    print("conversations", conversations)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        # image generation
        if self.data_args.gen_image_folder is not None:
            train_dataset = self.load_gen(self.data_args.gen_image_folder)
            for _ in range(self.data_args.gen_repeat):
                list_data_dict.append(train_dataset)
            print(f"finish loading gen image {len(train_dataset)}")
        
        # image editing
        if self.data_args.edit_image_folder is not None:
            train_dataset = self.load_edit(self.data_args.edit_image_folder)
            for _ in range(self.data_args.edit_repeat):
                list_data_dict.append(train_dataset)
            print(f"finish loading edit image {len(train_dataset)}")

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

        type_list = self.list_data_dict["type"]
        self.type_to_indices = {}
        for idx, t in enumerate(type_list):
            self.type_to_indices.setdefault(t, []).append(idx)
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def load_gen(self, gen_path):
        data_files = glob.glob(os.path.join(gen_path,'BLIP3o-Pretrain-Long-Caption', "*.tar")) + glob.glob(os.path.join(gen_path,'BLIP3o-Pretrain-Short-Caption', "*.tar")) + glob.glob(os.path.join(gen_path,'BLIP3o-Pretrain-JourneyDB', "*.tar"))
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128, cache_dir='../../cache_web')
        train_dataset = train_dataset.rename_column("jpg", "output_image")
        train_dataset = train_dataset.rename_column("txt", "input_prompt")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["output_image", "input_prompt", "type", "image_path"])])
        return train_dataset
    
    def load_edit(self, edit_path):
        data_files = glob.glob(os.path.join(edit_path, "*.tar"))
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128, cache_dir='../../cache_web')
        train_dataset = train_dataset.rename_column("input.jpg", "input_image")
        train_dataset = train_dataset.rename_column("output.jpg", "output_image")
        train_dataset = train_dataset.rename_column("txt", "input_prompt")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['TI2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["input_image", "output_image", "input_prompt", "type", "image_path"])])
        return train_dataset

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            try:
                sources = self.list_data_dict[i]
            except Exception as e:
                print(f"data error", i)
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue
            use_und = False
            if 'input_image' in sources and sources['input_image'] is not None:
                use_und = True
                if random.random() > 0.1:
                    sources["conversations"] = [
                        {"from": "human", "value": f"Edit the image: {sources['input_prompt']}\n<image>"},
                        {"from": "gpt", "value": "<image>"},
                    ]
                else:
                    # for cfg
                    sources["conversations"] = [
                            {"from": "human", "value": f"Edit the image.\n<image>"},
                        {"from": "gpt", "value": "<image>"},
                    ]
            else:
                use_und = False
                if random.random() > 0.1:
                    if sources['input_prompt'] is None or len(sources['input_prompt']) > 1000:
                        i = random.choice(self.type_to_indices[sources['type']])
                        continue
                    sources["conversations"] = [
                        {"from": "human", "value": f"Generate an image: {sources['input_prompt']}"},
                        {"from": "gpt", "value": "<image>"},
                    ]
                else:
                    # for cfg
                    sources["conversations"] = [
                        {"from": "human", "value": f"Generate an image."},
                        {"from": "gpt", "value": "<image>"},
                    ]

            if "output_image" in sources:

                def img_process(images, processor, image_aspect_ratio):
                    if image_aspect_ratio == "pad":

                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    else:
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    return images

                
                output_image = sources['output_image']
                # concat input and output, default last image is output image
                if use_und:
                    input_images = deepcopy(sources['input_image'])
                    all_images = [input_images]
                    all_images.append(output_image)
                else:
                    all_images = [output_image]

                all_pil_images = []

                for img in all_images:
                    try:
                        img = img.convert("RGB")
                        if img.size[0] == 1 or img.size[1] == 1:
                            print("wrong size", img.size)
                            all_pil_images = None
                            break
                        all_pil_images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        all_pil_images = None
                        break  # Skip to the next image if there's an error
                


                # If no valid images were found, randomly pick another item
                if all_pil_images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.choice(self.type_to_indices[sources['type']])
                    continue


                sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])
            data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "output_image" in self.list_data_dict[i]:
                process_images = img_process(
                    all_pil_images,
                    self.data_args.image_processor,
                    self.data_args.image_aspect_ratio,
                )
                if use_und:
                    data_dict["gen_image"] = process_images[-1:]
                    data_dict["und_image"] = process_images[:-1]
                else:
                    data_dict["gen_image"] = process_images

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - 257]
            label = label[: self.tokenizer.model_max_length - 257]
            i_s_pos.append(input_id.shape[0]+1)
            img_id = torch.full((257,), IMAGE_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = 151665
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((257,), IMAGE_TOKEN_IDX, dtype=label.dtype, device=label.device)
            img_label[0] = 151665
            label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch_gen_images = []
        batch_und_images = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])


        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"])  ## 1*1024*1176


        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
        else:
            batch["und_image"] = None
        batch["ids"] = ids

        batch["i_s_pos"] = i_s_pos

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    if data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError("Unknown data type. Please check the Dataloader type.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def unlock_vit(training_args, model_args, vision_tower):
    for n, p in vision_tower.named_parameters():
        p.requires_grad = True


def train(attn_implementation=None):

    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = UniLIP_InternVLForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    tokenizer = AutoProcessor.from_pretrained(model_args.mllm_hf_path).tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    if not model_args.fix_vit:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True 
        for p in model.get_model().multi_modal_projector.parameters():
            p.requires_grad = True 
    if not model_args.fix_llm:
        for p in model.get_model().language_model.parameters():
            p.requires_grad = True

    data_args.image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path).image_processor

    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    train_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param_names.append(name)
    print("trainable params", train_param_names)


    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.pretrain_path != 'none':
        pretrain_path = training_args.pretrain_path
        msg = model.load_state_dict(torch.load(pretrain_path), strict=False)
        print("load pretrain", pretrain_path)
        print(msg)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = NonMixTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    from tabulate import tabulate


    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

