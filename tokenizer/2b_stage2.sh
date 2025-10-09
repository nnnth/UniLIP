WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9997 --same_network \
    scripts/train_stage2.py config=configs/training/InternVL3_2B_DCAE/internvl3_2B_stage2.yaml \
    experiment.project="2B_stage2" experiment.name="2B_stage2" experiment.output_dir="2B_stage2" training.per_gpu_batch_size=16