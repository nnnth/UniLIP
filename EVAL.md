### Download Benchmarks
Download [WISE](https://github.com/PKU-YuanGroup/WISE.git) and [ImgEdit](https://huggingface.co/datasets/sysuyy/ImgEdit) and put them under `data` folder:
```
UniLIP
|──data
|────WISE
|────ImgEdit
```

### GenEval
In geneval_1b.sh, `MODEL` is the path to the model, and `CLS` is the class name of the model.
```
# UniLIP-1B
cd eval/geneval
bash geneval_1b.sh

# UniLIP-3B
cd eval/geneval
bash geneval_2b.sh
```
### WISE
```
# UniLIP-1B
cd eval/WISE
bash wise_1b.sh

# UniLIP-3B
cd eval/WISE
bash wise_2b.sh
```

### ImgEdit
```
# UniLIP-1B
cd eval/ImgEdit
bash imgedit_1b.sh

# UniLIP-3B
cd eval/ImgEdit
bash imgedit_2b.sh
```