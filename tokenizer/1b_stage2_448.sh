WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9997 --same_network \
    scripts/train_stage2.py config=configs/training/InternVL3_1B_DCAE/internvl3_1B_stage2_448.yaml \
    experiment.project="1B_stage2_448" experiment.name="1B_stage2_448" experiment.output_dir="1B_stage2_448" training.per_gpu_batch_size=16