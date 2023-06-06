## How to use LLM via an API server

Step 1: Request for GPU resources for setting up the API server
```bash
srun --gpus=1 --mincpus=8 --pty bash
```

Step 2: Activate conda environment
```bash
conda activate llm_lib
```

Step 3: CD to the root directory and run the API server
```bash
cd ../../../
python -m llm_lib.server --model_path /weka/Projects/local_llms/model_weights/vicuna13B --load_in_8bit 
```

Keep the HOST (e.g., 4gpu-01.ai...) and the PORT (default to 8000) of the server. These will be used in step 4


Step 4: **Open another bash client**, cd to this example direction, and run
```bash
conda activate llm_lib
python example.py --host http://HOST:PORT --sample_text "Answer the question: What do you know about the Applied Artificial Intelligence Institute?"
```

Where HOST and PORT need to be replaced with what you recorded in Step 3. HOST here is the full host, not the abbreviation. 
