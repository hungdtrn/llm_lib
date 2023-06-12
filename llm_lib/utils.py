
import re
import gc
import os
import time
import torch
import random
from pathlib import Path

import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer)
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

def load_model(model_path, load_in_8bit=False, wbits=0, groupsize=-1, auto_devices=False, 
               gpu_memory=None, cpu_memory=None, custom_class=None):
    model_path = os.path.realpath(model_path)
    model_dir = os.path.dirname(model_path)
    model_name = model_path.split("/")[-1]

    print(f"Loading {model_name}...")
    t0 = time.time()

    if custom_class is None:
        LoaderClass = AutoModelForCausalLM
    else:
        LoaderClass = custom_class
    
    if wbits > 0:
        from modules.GPTQ_loader import load_quantized

        model = load_quantized(model_dir, model_name,
                               wbits, groupsize=groupsize,
                               model_type="llama",
                               pre_layer=False,
                               custom_class=custom_class)
    else:
        params = {}
        
        
        if load_in_8bit and any((auto_devices, gpu_memory)):
            params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        elif load_in_8bit:
            params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

        if gpu_memory:
            memory_map = list(map(lambda x: x.strip(), gpu_memory))
            max_cpu_memory = cpu_memory.strip() if cpu_memory is not None else '99GIB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GIB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory
            params['max_memory'] = max_memory
        elif auto_devices:
            params["device_map"] = 'auto'
            total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            suggestion = round((total_mem - 3000) / 1000) * 1000
            if total_mem - suggestion < 800:
                suggestion -= 1000
            suggestion = int(round(suggestion / 1000))
            print(f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")

            max_memory = {}
            for i in range(torch.cuda.device_count()):
                _ = torch.tensor([0], device=i)
                max_memory[i] = f'{suggestion}GIB'
            max_memory['cpu'] = cpu_memory.strip() if cpu_memory is not None else '99GIB'

            params['max_memory'] = max_memory

        checkpoint = Path(model_path)
        
        # if load_in_8bit and params.get('max_memory', None) is not None and params['device_map'] == 'auto':
        if params.get('max_memory', None) is not None and params['device_map'] == 'auto':
            print("auto device")
            config = AutoConfig.from_pretrained(checkpoint)
            
            with init_empty_weights():
                model = LoaderClass.from_config(config)
                    
            model.tie_weights()
            params['device_map'] = infer_auto_device_map(
                model,
                dtype=torch.int8,
                max_memory=params['max_memory'],
                no_split_module_classes=model._no_split_modules
            )
            
        model = LoaderClass.from_pretrained(checkpoint, **params)
            
    # Load the tokenizer
    if type(model) is transformers.LlamaForCausalLM:
        tokenizer = LlamaTokenizer.from_pretrained(Path(model_path), clean_up_tokenization_spaces=True)
        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(model_path))
        
    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

