import os 
import inspect
import re
import sys
from pathlib import Path

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, str(Path(f"{dir_path}/repositories/GPTQ-for-LLaMa")))
import llama_inference_offload

try:
    from modelutils import find_layers
except ImportError:
    from utils import find_layers

try:
    from quant import make_quant
    is_triton = False
except ImportError:
    import quant
    is_triton = True


# This function is a replacement for the load_quant function in the
# GPTQ-for_LLaMa repository. It supports more models and branches.
def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128, eval=True,
                custom_class=None):

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    
    if custom_class is None:
        model = AutoModelForCausalLM.from_config(config)
    else:
        print(config)
        model = custom_class.from_config(config)
        
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    if not is_triton:
        gptq_args = inspect.getfullargspec(make_quant).args

        make_quant_kwargs = {
            'module': model,
            'names': layers,
            'bits': wbits,
        }
        if 'groupsize' in gptq_args:
            make_quant_kwargs['groupsize'] = groupsize
        if 'faster' in gptq_args:
            make_quant_kwargs['faster'] = faster_kernel
        if 'kernel_switch_threshold' in gptq_args:
            make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

        make_quant(**make_quant_kwargs)
    else:
        quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if is_triton:
        raise Exception

    model.seqlen = 2048
    print('Done.')

    return model


# Used to locate the .pt/.safetensors quantized file
def find_quantized_model_file(model_dir, model_name, wbits, groupsize):
    path_to_model = Path(f'{model_dir}/{model_name}')
    pt_path = None
    priority_name_list = [
        Path(f'{model_dir}/{model_name}{hyphen}{wbits}bit{group}{ext}')
        for group in ([f'-{groupsize}g', ''] if groupsize > 0 else [''])
        for ext in ['.safetensors', '.pt']
        for hyphen in ['-', f'/{model_name}-', '/']
    ]
    for path in priority_name_list:
        if path.exists():
            pt_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .pt
    # or the last .safetensors found in its folder as a last resort
    if not pt_path:
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .pt model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_safetensors[-1]

    return pt_path


# The function that loads the model in modules/models.py
def load_quantized(model_dir, model_name, wbits, groupsize, model_type=None, pre_layer=True, custom_class=None):

    # Find the model type
    if model_type is None:
        name = model_name.lower()
        if any((k in name for k in ['llama', 'alpaca', 'vicuna'])):
            model_type = 'llama'
        elif any((k in name for k in ['opt-', 'galactica'])):
            model_type = 'opt'
        elif any((k in name for k in ['gpt-j', 'pygmalion-6b'])):
            model_type = 'gptj'
        else:
            print("Can't determine model type from model name. Please specify it manually using --model_type "
                  "argument")
            exit()
    else:
        model_type = model_type.lower()

    # Select the appropriate load_quant function
    if pre_layer and model_type == 'llama':
        load_quant = llama_inference_offload.load_quant
    elif model_type in ('llama', 'opt', 'gptj'):
        if pre_layer:
            print("Warning: ignoring --pre_layer because it only works for llama model type.")
        load_quant = _load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama', 'opt' and 'gptj' are supported")
        exit()

    # Find the quantized model weights file (.pt/.safetensors)
    path_to_model = Path(f'{model_dir}/{model_name}')
    pt_path = find_quantized_model_file(model_dir, model_name, wbits, groupsize)
    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()
    else:
        print(f"Found the following quantized model: {pt_path}")

    # qwopqwop200's offload
    if model_type == 'llama' and pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, pre_layer, custom_class=custom_class)
    else:
        threshold = False if model_type == 'gptj' else 128
        model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, kernel_switch_threshold=threshold, custom_class=custom_class)

        # accelerate offload (doesn't work properly)
        if torch.cuda.device_count() > 1:
            max_memory = accelerate.utils.get_balanced_memory(model)

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
            print("Using the following device map for the quantized model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        else:
            model = model.to(torch.device('cuda:0'))

    return model
