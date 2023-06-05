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


Step 4: **Open another bash client**, cd to this example direction, and run
```bash
conda activate llm_lib
python example.py --host SERVER_HOST --sample_text "Answer the question: What do you know about the Applied Artificial Intelligence Institute?"
```