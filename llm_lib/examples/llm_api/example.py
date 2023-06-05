from llm_lib.client import LLMClient
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--sample_text", default="Answer the question: What is A2I2?")
    args = parser.parse_args()
    client = LLMClient(host=args.host)
    
    print("-------------")
    print("Sample text:", args.sample_text)
    output = client.create_completion([args.sample_text])
    print("Output:", output.response.choices[0].text)

