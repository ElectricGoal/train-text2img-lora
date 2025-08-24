# train_text2img_lora

## 1. Create environment

```
conda env create -f environment.yml
conda activate train_ipadapter
```

## 2. Create caption for dataset

```
python create_caption.py
```

## 3. Training

```
bash run_train.sh
```