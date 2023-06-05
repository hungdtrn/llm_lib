### This  implementation is inspired by python-llama-cpp (https://github.com/abetlen/llama-cpp-python)

import os
import json
from typing import List, Optional, Literal, Union, Iterator, Dict
from typing_extensions import TypedDict
import uuid
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field, create_model_from_typeddict
from llm_lib.model import LLMModel
from .request_body import CompetionRequestBody
from .response_body import CompletionRepsoneBody


def main(model_path, load_in_8bit=True, wbits=0, groupsize=-1,
         auto_devices=False):
    app = FastAPI(
        title="API Wrapper for Local LLMs",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    model = LLMModel(model_path, load_in_8bit=load_in_8bit, wbits=wbits, groupsize=groupsize, 
                     auto_devices=auto_devices)

    @app.post(
        "/v1/completions",
        response_model=CompletionRepsoneBody,
    )
    def create_completion(request: CompetionRequestBody):
        data = request.dict()
        created = int(time.time())
        
        output_str, _ = model.generate(
            prompts=data.pop("prompt"),
            **data
        )
        
        output_str = [x for y in output_str for x in y]

        completion_or_chunks = {
            "id":  f"cmpl-{str(uuid.uuid4())}",
            "object": "text_completion",
            "created": created,
            "model": 'local',
            "choices": [
                {
                    "text": output,
                    "index": i,
                    "finish_reason": None,
                }
            for i, output in enumerate(output_str)],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": [0 for _ in output_str],
                "total_tokens": [0 for x in output_str],
            },
        }

                
        completion: CompletionRepsoneBody = completion_or_chunks
        return completion
    
    return app


if __name__ == "__main__":
    import os
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the model weight", required=True)
    parser.add_argument("--wbits", help="Number of quantization bits", default=0, type=int)
    parser.add_argument("--groupsize", help="Groupsize parameter of the GPTQ quantization method", default=-1, type=int)
    parser.add_argument("--load_in_8bit", help="Whether to load 8 bit", action='store_true', default=False)
    parser.add_argument("--auto_devices", help="Whether to auto-distribute to gpu devices", action='store_true', default=False)

    args = parser.parse_args()
    
    app = main(args.model_path, args.load_in_8bit, args.wbits, args.groupsize, args.auto_devices)

    uvicorn.run(
        app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )

