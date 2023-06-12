import os
import logging
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import fire
import torch
import loguru
import deepspeed
import numpy as np
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer, LlamaTokenizer, LlamaForSequenceClassification)
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.utils import see_memory_usage
 
from dataset import load_dataset

dataset = load_dataset("yelp_review_full")
print(dataset["train"][100])
logger = loguru.logger

def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')

def collate_function(batch):
    out_dict = {}
    for b in batch:
        for k in b.keys():
            if k not in out_dict:
                out_dict[k] = []
                
            out_dict[k].append(b[k])
    
    for k in out_dict.keys():
        out_dict[k] = torch.stack(out_dict[k])
    
    return out_dict




def train(checkpoint_path = "/weka/Projects/local_llms/model_weights/vicuna13B", 
          log_every=5, local_rank=-1):
    try:
        model = LlamaForSequenceClassification.from_pretrained(checkpoint_path, num_labels=5)
        for param in model.model.parameters():            
            param.requires_grad = False

        tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.pad_token_id = 0
    except:
        model = AutoModel.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset["train"] = dataset["train"].select(range(1000))
    dataset["test"] = dataset["test"].select(range(1000))

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    batch_size = 8
     
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_function)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size, collate_fn=collate_function)
    
    
    device = (torch.device("cuda", local_rank) if (local_rank > -1)
              and torch.cuda.is_available() else torch.device("cpu"))

    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 5e-5
            }
        },
        "bf16": {
            "enabled": True, 
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        }
    }

    model, _, _, _ = deepspeed.initialize(model=model,
                                            model_parameters=model.parameters(),
                                            config=ds_config)
    model.gradient_checkpointing_enable()
    
    log_dist("DeepSpeed engine created", ranks=[0], level=logging.INFO)
    log_dist(
        f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}",
        ranks=[0],
        level=logging.INFO)
    log_dist(
        f"Total number of training parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,d}",
        ranks=[0],
        level=logging.INFO)

    model.train()
    losses = []
    start_step = 0
    num_iterations = 10
    for step, batch in enumerate(train_dataloader, start=start_step):
        if step >= num_iterations:
            break
        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to(device)
        
        loss = model(**batch)[0]

        # Backward pass
        model.backward(loss)
        
        # Optimizer Step
        model.step()
        losses.append(loss.item())

        if step % log_every == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                     ranks=[0],
                     level=logging.INFO)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    fire.Fire(train)
