import json
from urllib import response
import requests
import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

class ResponseObj(dict):
    def __init__(self, resp: dict) -> None:
        for k, v in resp.items():
            self.__setattr__(k, converto_to_response_obj(v))
    
    # def __setattr__(self, __name: str, __value) -> None:
    #     return super().__setattr__(__name, __value)
    
    # def __getattr__(self, k):
    #     return self[k]
    

def converto_to_response_obj(resp):
    if isinstance(resp, dict) and not isinstance(resp, ResponseObj):
        return ResponseObj(resp)
    elif isinstance(resp, list):
        return [converto_to_response_obj(x) for x in resp]
    else:
        return resp

class PromptTemplate:
    Alpaca ='alpaca'
    GPT4Chan = 'gpt4chan'
    OpenAssistant =  'open_assistant'
    QA = 'qa'
    Vicuna = 'vicuna'
    GPT4All = "alpaca"
    
            

class LLMClient:
    def __init__(self, host="http://0.0.0.0:8000/v1") -> None:
        self.prefix = "v1/completions"
        self.template = self.load_template()
        self.host = host 
        
    @classmethod
    def cls_process_prompt(cls, prompt, input=None):
        template = cls.cls_load_template()
        if input is None:
            return [template["prompt_no_input"].format(prompt=x) for x in prompt]
        else:
            return [template["prompt_input"].format(prompt=x, input=y) for (x, y) in zip(prompt, input)]

    @classmethod
    def cls_load_template(cls):
        template_name = PromptTemplate.Alpaca
            
        with open(os.path.join(dir_path, "prompt_templates/{}.json".format(template_name))) as f:
            template = json.load(f)
        
        return template


    def get_model_name(self):
        prefix = self.prefix.replace("completions", "model_name")
        response = requests.get("{}/{}".format(self.host, prefix)).json()
        name = None
        try:
            name = response["name"].lower()
        except Exception as e:
            print("Error when getting model name", e)
        
        return name
            
    def load_template(self):
        template_name = PromptTemplate.Alpaca
            
        with open(os.path.join(dir_path, "prompt_templates/{}.json".format(template_name))) as f:
            template = json.load(f)
        
        return template
            
    def process_prompt(self, prompt, input=None):
        if input is None:
            return [self.template["prompt_no_input"].format(prompt=x) for x in prompt]
        else:
            return [self.template["prompt_input"].format(prompt=x, input=y) for (x, y) in zip(prompt, input)]
        
    def process_response(self, response):
        return {"text": response.split(self.template["response_split"])[-1]}
    
    def create_completion(self, prompt, input=None, max_new_tokens=200,
            do_sample=True,
            temperature=0.5,
            top_p=1,
            typical_p=1,
            repetition_penalty=1.1,
            encoder_repetition_penalty=1,
            top_k=0,
            min_length=0,
            no_repeat_ngram_size=0,
            num_beams=1,
            penalty_alpha=0,
            length_penalty=1,
            early_stopping=False,
            seed=-1,
            add_bos_token=True,
            custom_stopping_strings='',
            truncation_length=2048,
            ban_eos_token=False,
            skip_special_tokens=True,
            stopping_strings=[],
            output_scores=False,
            num_return_sequences=1,
            **kwargs):
        
        
        request_body = json.dumps({ 'prompt': self.process_prompt(prompt),
                                    'max_new_tokens': max_new_tokens,
                                    'do_sample': do_sample,
                                    'temperature': temperature,
                                    'top_p': top_p,
                                    'typical_p': typical_p,
                                    'repetition_penalty': repetition_penalty,
                                    'encoder_repetition_penalty': encoder_repetition_penalty,
                                    'top_k': top_k,
                                    'min_length': min_length,
                                    'no_repeat_ngram_size': no_repeat_ngram_size,
                                    'num_beams': num_beams,
                                    'penalty_alpha': penalty_alpha,
                                    'length_penalty': length_penalty,
                                    'early_stopping': early_stopping,
                                    'seed': seed,
                                    'add_bos_token': add_bos_token,
                                    'custom_stopping_strings': custom_stopping_strings,
                                    'truncation_length': truncation_length,
                                    'ban_eos_token': ban_eos_token,
                                    'skip_special_tokens': skip_special_tokens,
                                    'stopping_strings': stopping_strings,    
                                    'output_scores': output_scores,           
                                    'num_return_sequences': num_return_sequences, 
                                })
        response = requests.post("{}/{}".format(self.host, self.prefix), data=request_body)    
        
        print(response.json())

        if response.status_code != 200:
            print(response.json())
            raise Exception(response)
        else:
            response = response.json()
            for i in range(len(response["choices"])):
                response["choices"][i].update(self.process_response(response["choices"][i]["text"]))

            return converto_to_response_obj({"response": response})
            
    
        