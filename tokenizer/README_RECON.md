### Environment
```
conda create -n UniLIP_recon python=3.8
conda activate UniLIP_recon
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Training
We use the same data as the generation task: BLIP3o-Pretrain. Please refer to the [README](https://github.com/nnnth/UniLIP/blob/main/README.md) for more details.

The training process is divided into three stages:
Stage 1: At a 224x224 resolution, only the decoder is trained.
```
bash 1b_stage1.sh
```

Stage 2: At a 224x224 resolution, all modules are trained. You need to set the `stage1_ckpt` parameter in the config file to the path of the checkpoint from Stage 1.
```
bash 1b_stage2.sh
```

Stage 3: At a 448x448 resolution. Similarly, you need to set the `stage1_ckpt` parameter in the config to the path of the checkpoint from the previous stage.
```
bash 1b_stage2_448.sh
```


### Evaluation
The pretrained UniLIP (CLIP+Pixel Decoder) checkpoints can be downloaded in [repo](https://huggingface.co/kanashi6/UniLIP).
#### ImageNet-1k val
Download [imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) to `data` folder then convert it to webdataset format by:
```
cd data
python convert_imagenet_to_wds.py imagenet_sharded
```

Then run the evaluation script, specify the checkpoint by `checkpoint_path`:
```
bash 1b_test.sh
```

#### Understanding Benchmarks
We use VLMEvalKit for evaluation. Taking MMVP as an example, you need to set `ckpt_path` in VLMEvalkit/mmvp.sh to trained UniLIP weights, and then execute the mmvp.sh script. For instructions on testing other benchmarks, please refer [here](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md).


### Reconstruction Demo
In 1b_inference.sh, set `checkpoint_path` to the UniLIP weights and `img_path` to the input image. Then, run the following:

```
bash 1b_inference.sh
```
The output image will be saved as recon.jpg
