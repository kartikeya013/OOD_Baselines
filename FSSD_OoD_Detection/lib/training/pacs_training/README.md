# Train MNIST for OoD detection


```bash
python main.py --rotation <angle>
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

Resume the training with `python main.py --resume --lr=0.01 --rotation 45`
