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
from .utils import load_model

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
print(os.path.dirname(os.path.realpath(__file__)))

def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def get_max_prompt_length(truncation_length, max_new_tokens):
    max_length = truncation_length - max_new_tokens
    return max_length


def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

# Copied from https://github.com/PygmalionAI/gradio-ui/
class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]

            for i in range(len(self.sentinel_token_ids)):
                # Can't unfold, output is still too tiny. Skip.
                if trimmed_sample.shape[-1] < self.sentinel_token_ids[i].shape[-1]:
                    continue
                for window in trimmed_sample.unfold(0, self.sentinel_token_ids[i].shape[-1], 1):
                    if torch.all(torch.eq(self.sentinel_token_ids[i][0], window)):
                        return True
        return False


class LLMModel:
    def __init__(self, model_path, load_in_8bit=False, wbits=0, groupsize=128, auto_devices=False, 
               gpu_memory=None, cpu_memory=None, custom_class=None) -> None:
        self.model, self.tokenizer = load_model(model_path,load_in_8bit=load_in_8bit,
                                      wbits=wbits, groupsize=groupsize,
                                      gpu_memory=gpu_memory,
                                      cpu_memory=cpu_memory,
                                      auto_devices=auto_devices,
                                      custom_class=custom_class)

    def prepare_data(self, prompts, add_special_tokens, add_bos_token, truncation_length=2048, max_new_tokens=128, 
                     contexts=None, context_split="", num_context_tokens=1, fuse_layer=1):
        if contexts is None:
            input_ids, attention_mask = self.encode(prompts, add_special_tokens=add_special_tokens, add_bos_token=add_bos_token, truncation_length=get_max_prompt_length(truncation_length, max_new_tokens))

            data_dict = {'inputs': input_ids, "attention_mask": attention_mask, "num_tokens": len(input_ids[0])}
        else:
            input_embeds, attention_mask, context_mask = self.add_context_to_prompts(prompts, contexts, context_split, 
                                                                                     add_special_tokens=add_special_tokens, 
                                                                                     add_bos_token=add_bos_token, 
                                                                                     num_context_tokens=num_context_tokens,
                                                                                     fuse_layer=fuse_layer)

            data_dict = {'inputs_embeds': input_embeds, "attention_mask": attention_mask,  
                         "context_mask": context_mask, "fuse_layer": fuse_layer, "num_tokens": len(input_embeds[0])}

        return data_dict
    
    def encode_context(self, context, add_special_tokens, add_bos_token, num_context_tokens=1, fuse_layer=1, **kwargs):
        with torch.no_grad():
            context_kwargs = kwargs.copy()
            context_kwargs.update({
                "output_hidden_states": True,
                "max_new_tokens": 1
            })
            
            _, out = self.generate(context, add_special_tokens, add_bos_token,
                                   **context_kwargs)

            hidden_states = out.hidden_states[0][fuse_layer][:, -1-num_context_tokens:-1, :]
            return hidden_states
        
    def add_context_to_prompts(self, prompts, contexts, context_split, add_special_tokens, add_bos_token, num_context_tokens=1, fuse_layer=1, **kwargs):
        parts_str = [[] for _ in range(len(contexts) + 1)]
        masks = []
        
        # Split the inputs to parts
        for inp in prompts:
            current_parts = inp.split(context_split)

            for i, p in enumerate(current_parts):
                parts_str[i].append(p)
                
        assert len(parts_str) == len(contexts) + 1
                
        # Embed the context
        contexts = self.encode_context(contexts, add_special_tokens=add_special_tokens, add_bos_token=add_bos_token, 
                                       num_context_tokens=num_context_tokens, fuse_layer=fuse_layer, **kwargs)
                
        # Embed the splited text
        parts_embed = []
        all_inps = []
        start, end = 0, 0
        idx_list = []
        for p in parts_str:
            inps, m = self.encode(p, add_special_tokens=add_special_tokens, add_bos_token=add_bos_token)
            masks.append(m)
            
            end = end + inps.shape[1]
            all_inps.append(inps)
            idx_list.append((start, end))
            start = end
            
        all_embed = self.model.model.embed_tokens(torch.cat(all_inps, dim=1))

        for (start, end) in idx_list:
            parts_embed.append(all_embed[:, start:end])
            
        dummy_mask = torch.ones_like(masks[0])[:, 0:1]
        out_input_ids, out_masks = [parts_embed[0]], [masks[0]]
        context_masks = [torch.zeros_like(masks[0])]
        for i, context in enumerate(contexts):
            out_input_ids = out_input_ids + [context.repeat(len(prompts), 1, 1), parts_embed[i+1]]
            out_masks = out_masks + [dummy_mask.clone(), masks[i+1]]
            context_masks = context_masks + [dummy_mask.clone(), torch.zeros_like(masks[i+1])]
            
            
        out_input_ids = torch.cat(out_input_ids, dim=1)
        out_masks = torch.cat(out_masks, dim=1)
        context_masks = torch.cat(context_masks, dim=1)

        assert out_input_ids.size(1) == out_masks.size(1)
        
        return out_input_ids, out_masks, context_masks

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        out = self.tokenizer(prompt, padding=True, return_tensors='pt', add_special_tokens=add_special_tokens)
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == self.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
            

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
            
        
        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]
            attention_mask = attention_mask[:, -truncation_length:]

        return input_ids.cuda(), attention_mask.cuda()
    
    def decode(self, output_ids, skip_special_tokens=True):
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
        
        if skip_special_tokens:
            output = [x.replace(r'<|endoftext|>', '') for x in output]
        
        return output
        
    def generate(self, prompts, add_special_tokens=True, add_bos_token=True, 
                 skip_special_tokens=True, truncation_length=2048, max_new_tokens=128, 
                 stops=[], contexts=None, context_split="", num_context_tokens=1, fuse_layer=1, 
                 decode=True, seed=-1, **kwargs):
        clear_torch_cache()
        seed = set_manual_seed(seed)
        t0 = time.time()

        # Encode the prompt
        data_dict = self.prepare_data(prompts, add_special_tokens=add_special_tokens,
                                      add_bos_token=add_bos_token,
                                      truncation_length=truncation_length, max_new_tokens=max_new_tokens,
                                      contexts=contexts, context_split=context_split,
                                      num_context_tokens=num_context_tokens,
                                      fuse_layer=fuse_layer)
        
        num_input_tokens = data_dict.pop("num_tokens")
        
        eos_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        
        # set the stopping criterias
        stopping_criteria_list = transformers.StoppingCriteriaList()
        for st in [stops]:
            if type(st) is list and len(st) > 0:
                sentinel_token_ids = [self.encode(string , add_special_tokens=add_special_tokens, add_bos_token=add_special_tokens)[0] for string in st]
                stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=num_input_tokens))

        # prepare the params for generation
        params = {"return_dict_in_generate": True, 
                  "max_new_tokens": max_new_tokens,
                  "stopping_criteria": stopping_criteria_list,
                  "eos_token_id": eos_token_ids}
        params.update(data_dict)
        
        
        default_generation_config_keys = ["max_length", "max_new_tokens", "min_length", "min_new_tokens", "early_stopping", "max_time",
                                    "do_sample", "num_beams", "num_beam_groups", "penalty_alpha", "use_cache", 
                                    "temperature", "top_k", "top_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
                                    "diversity_penalty", "repetition_penalty", "encoder_repetition_penalty", "length_penalty",
                                    "no_repeat_ngram_size", "bad_words_ids", "force_words_ids", "renormalize_logits", "constraints",
                                    "forced_bos_token_id", "forced_eos_token_id", "forced_eos_token_id", "exponential_decay_length_penalty",
                                    "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids",
                                    "num_return_sequences", "output_attentions", "output_hidden_states", "output_scores",
                                    "pad_token_id", "bos_token_id", "eos_token_id", "encoder_no_repeat_ngram_size", "decoder_start_token_id",]
        for k in default_generation_config_keys:
            if k in kwargs:
                params[k] = kwargs[k]
        
        with torch.no_grad():
            outputs = self.model.generate(**params)
            sequences = outputs.sequences.cuda()
            new_tokens = sequences.shape[-1]
            
            sequences = self.decode(sequences, skip_special_tokens)
            output_str = []
            if decode:
                num_returned_sequences = params.get("num_return_sequences", 1)

                for i in range(len(prompts)):
                    current_output = []
                    for j in range(num_returned_sequences):
                        current_output.append(sequences[i * num_returned_sequences + j])
                    output_str.append(current_output)
                
            clear_torch_cache()
            t1 = time.time()
            print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {num_input_tokens}, seed {seed})')        
            
            return output_str, outputs
        