### Dataset Preparation
#### Pretraining Dataset
For generation, we use BLIP3-o Pretrain: [BLIP3o-Pretrain-Long-Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [BLIP3o-Pretrain-Short-Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption) and [BLIP3o-Pretrain-JourneyDB](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB). For editing, we use [GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M). Please download them to the `data` directory. For format consistency, we convert `GPT-Image-Edit` to the webdataset format:

```
cd data
python convert_gptedit.py
```


#### SFT Dataset
We use [BLIP3o-60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) and [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image). After downloading the two repositories into the `data` directory, run the following command to unify the format.

```
cd data
bash untar_sharegpt.sh
bash prepare_gen_sft.sh
python convert_janus4o_ti2i.py
```
This will generate two folders, `gen_sft` and `edit_sft`.

### Download Pretraining Weight
For the reconstruction training, we use the InternViT from [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) and [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) as the CLIP respectively, and [DC-AE-f32c32](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) as the pixel decoder. For the generation training, we use [SANA-0.6B](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers) and [SANA-1.6B](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers) as the DiT respectively. If you cannot access Hugging Face directly, you can download these repositories locally and then update the path in the training script to the local absolute path.

For generation and editing training, you also need to download the reconstruction-trained UniLIP weights. Please download them from [here](https://huggingface.co/kanashi6/UniLIP), and then place the weights into the tokenizer_ckpt directory.
```
UniLIP
|──tokenizer_ckpt
|────1b_unilip.pth
|────2b_unilip.pth
```

### Training
Open the `scripts` folder first:

1B Stage1, connector training

```
bash run_unilip_1b_stage1.sh
```

1B Stage2, Pretraining

```
bash run_unilip_1b_stage2.sh
```
1B Stage3, sft

```
bash run_unilip_1b_stage3.sh
```

2B training is similar to 1B.