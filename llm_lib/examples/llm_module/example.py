import argparse
from llm_lib.utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the model weight", required=True)
    parser.add_argument("--wbits", help="Number of quantization bits", default=0, type=int)
    parser.add_argument("--groupsize", help="Groupsize parameter of the GPTQ quantization method", default=-1, type=int)
    parser.add_argument("--load_in_8bit", help="Whether to load 8 bit", action='store_true', default=False)
    parser.add_argument("--auto_devices", help="Whether to auto-distribute to gpu devices", action='store_true', default=False)
    parser.add_argument("--sample_text", default="Answer the question: What is A2I2?")

    args = parser.parse_args()
    transformer, tokenizer = load_model(model_path=args.model_path,
                                        load_in_8bit=args.load_in_8bit,
                                        wbits=args.wbits,
                                        groupsize=args.groupsize,
                                        auto_devices=args.auto_devices)

    print("Sample text: ", args.sample_text)
    params = {
        "return_dict_in_generate": True,
        "max_new_tokens": 20
    }
    encoding = tokenizer(args.sample_text, return_tensors='pt')
    for k in encoding.keys():
        params[k] = encoding[k].to(transformer.device)
        
    outputs = transformer.generate(**params)
    sequences = outputs.sequences.cuda()[0]
    sequences = tokenizer.decode(sequences)
    
    print("Output:", sequences)
    