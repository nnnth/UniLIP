#!/bin/bash



MODEL="../../work_dirs/2b_stage3/checkpoint-xxx"
CLS_NAME="UniLIP_InternVLForCausalLM"


# Total number of GPUs/chunks.
N_CHUNKS=4

# Launch processes in parallel for each GPU/chunk.
for i in $(seq 0 $(($N_CHUNKS - 1))); do
    echo "Launching process for GPU $i (chunk index $i of $N_CHUNKS)"
    CUDA_VISIBLE_DEVICES=$i python wise.py --cls "$CLS_NAME" --model "$MODEL" --index $i --n_chunks $N_CHUNKS &
done

# Wait for all background processes to finish.
wait
echo "All background processes finished."


