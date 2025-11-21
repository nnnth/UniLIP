python extract_vit.py --ckpt_path "checkpoint-xxx/ema_model/pytorch_model.bin" \
    --output_path "./unilip_vit.pth"
torchrun --nproc-per-node=4 run.py --data MMVP --model UniLIP-1B --verbose