# SLiYME
Official Repository for SLiYME Project @DL&amp;NLP

## Requirements 
Before you begin, make sure to install the required dependencies :
```bash
pip install -r requirements.txt 
```

## Fine-tuning Llama for Lyric Generation 
For fine-tuning, you can follow the code below. 
```bash
python train_refac.py 
```

For evaluation, you can follow the code below.
```bash
python evaluate_w_rhyme.py
```

## Convention
```bash
S-LiYME/
├── data/
│   ├── train.json
│   └── val.json
├── models/
│   └── loss.py
├── utils/
│   └── utils.py
├── phonetic-word-embedding/
├── scripts/
│   ├── train_refac.py
│   └── evaluate_w_rhyme.py
├── simvecs/
├── requirements.txt
├── LICENSE
└── README.md
```
