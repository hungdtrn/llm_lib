## How to use LLM as a module on SLURM cluster

Step 1: Request for GPU resources
```bash
srun --gpus=1 --mincpus=8 --pty bash
```

Step 2: Activate conda environment
```bash
conda activate llm_lib
```

Step 3: Run the python example file
```bash
python example.py --model_path /weka/Projects/local_llms/model_weights/vicuna13B --load_in_8bit --sample_text "Answer the question: What do you know about the Applied Artificial Intelligence Institute?"
```