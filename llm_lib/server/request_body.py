from optparse import Option
from typing import List, Optional, Literal, Union, Iterator, Dict

from pydantic import BaseModel

class CompetionRequestBody(BaseModel):
    prompt: List[str]
    model: Optional[str]
    suffix: Optional[str]
    max_tokens: Optional[int]=16
    temperature: Optional[float]=0.1
    top_p: Optional[float]
    n: Optional[int]=1
    stream: Optional[bool]=False
    logprobs: Optional[int]
    echo: Optional[bool]=False
    stop: Optional[List[str]]=[]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    best_of: Optional[int]
    logit_bias: Optional[Dict]
    seed: Optional[int]=-1
    
    ## Hugging face params
    top_k: Optional[int]
    num_beams: Optional[int]=1
    use_cache: Optional[bool]=True
    repetition_penalty: Optional[float]
    num_return_sequences: Optional[int]=1
    output_scores: Optional[bool]=False
    do_sample: Optional[bool]=True
    min_length: Optional[int]
    no_repeat_ngram_size: Optional[int]
    penalty_alpha: Optional[float]
    length_penalty: Optional[float]
    early_stopping: Optional[bool]=False
    output_scores: Optional[bool]

    ## Custom param
    add_bos_token: Optional[bool]=True
    ban_eos_token: Optional[bool]=False
    skip_special_tokens: Optional[bool]=True
    custom_stopping_strings: Optional[str]=""
    truncation_length: Optional[int]=2048
    max_new_tokens: Optional[int] = 200
    stopping_strings: Optional[List[str]]
    
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": ["What is the capital of France?"],
                "stop": ["\n", "###"],
            }
        }

class CompetionWithExampleRequestBody(CompetionRequestBody):
    contexts: List[str]
    context_split: str
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": ["Answer the question:\n<ContextHere>\nQuestion: a + b=?\nAnswer:"],
                "contexts": ["Context: a=1, b=2"],
                "context_split": "<ContextHere>"
            }
        }
