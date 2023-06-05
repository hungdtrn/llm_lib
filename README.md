# LLM library for loading and using LLM models
This library serves as a platform for utilizing and creating applications based on pre-existing foundation models. Its features include:    
1. Loading large language models (LLMs) as PyTorch modules.
2. Establishing an API server that resembles the ChatGPT API.

Illustrative examples for each use case can be found in the example folder.
## 1. Installation

Use the package manager [conda](https://conda.io/projects/conda/en/latest/index.html) to install the required parameters

```bash
conda env create -f env.yml
```

Install the dependencies of the module (GPTQ)
```bash
cd llm_lib/modules/repositories/
git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda
cd GPTQ-for-LLaMa
python setup_cuda.py install
```

After installing GPTQ, navigate back the the root folder 
```bash
cd ../../../../
```

Add the library to PYTHONPATH via
```bash
pip install -e .
```

## 2. Usage
The library supports two main use cases:
1. Loading LLMs and utilizing them as regular PyTorch modules. This is ideal for users seeking complete control over the model.
2. Establishing an API server resembling the ChatGPT API and employing an API client to connect to the API. This is suitable for users who prefer not to modify their model code. 

### 2.1. Use LLM as Pytorch Module
```bash
from llm_lib.utils import load_model

transformer, tokenizer = load_model(model_path, **kwargs)
```

Supported parameters for `load_model`
| Parameter | Description |
| --- | --- |
| model_path | Path to the downloaded model weights. (For A2I2 students/researchers, please refer to the Supported Pre-trained Weights section) |
| load_in_8bit | Determines whether to load the model with 8-bit precision. This option allows for loading models using fewer GPUs, with a slight tradeoff in performance. |
| auto_devices | Controls whether the GPU usage is distributed across multiple GPUs automatically. |
| wbits, groupsize | Parameters for utilizing GPTQ quantization. For more details, please refer to [this paper](https://arxiv.org/abs/2210.17323) and [this repository](https://github.com/qwopqwop200/GPTQ-for-LLaMa). |

Examples are provided in `examples/`

### 2.2. Use LLM via API

### Step 1: Setting up the API server
By default the APIs will be accessed via "http://0.0.0.0:8000/v1/". The documentation of the APIs is accessed via "http://0.0.0.0:8000/v1/docs#"

```
python -m llm_lib.server --model_path PATH_TO_MODEL_WEIGHT
```

Supported parameters are similar to those in `load_model` above


### Step 2: Use the LLMClient
Example of using the LLMClient in the python code

```
from llm_lib.client import LLMClient

local_llm = LLMClient(host="http://0.0.0.0:8000/v1")

# Sentence completion
completion = local_llm.create_completion(prompt="Hello, How are you?", max_tokens=128, temperature=1.0).response.choices[0].text.strip()
```

**Note:** If you host the API server at a different machine with a different address, you need to replace "http://0.0.0.0:8000/v1" with your address.

Examples are provided in examples/

## 3. Download model
You can automatically download a model from HuggingFace (HF) using `download_model.py`
```bash
python download-model.py organization/model
```

For example
```bash
python download-model.py facebook/opt-1.3b
```

## 4. Supported Pre-trained Weights
A2I2 students and researchers can utilize the downloaded model weights stored in `/weka/Projects/local_llms/model_weights/`. It is important to note that models ending with `4bits-128g` or `4bits` require specific flags to be enabled during execution.
For `*-4bits-128g` models, they should be executed with the flags `--wbits 4 --groupsize 128`.
For `*-4bits` models, only the `--wbits 4` flag needs to be used."

Example
```bash

# As pytorch module
transformer, tokenizer = load_model(model_path="/weka/Projects/local_llms/model_weights/TheBloke_vicuna-13B-1.1-GPTQ-4bit-128g/", wbits=4, groupsize=128)

# As API
python -m llm_lib.server --model_path /weka/Projects/local_llms/model_weights/TheBloke_vicuna-13B-1.1-GPTQ-4bit-128g/ --wbits 4 --groupsize 128
```

## 5. Documentation
The proper documation will be written soon.

## 6. Resources
Some of the codes are borrowed from
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- [GPTQ-for-llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [Huggingface](https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/llama#transformers.LlamaForCausalLM)

