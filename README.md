# S:LiYME: Scores for Lyrics in Rhyme
Official Repository for SLiYME Project @DL&amp;NLP
---
## Introduction
S-LiYME generates song lyrics by considering syllable structure, context consistency, and rhyme.

## Code Structure
- `data/`: dataset for training and evaluation
- `models/`: model related files
- `utils/`: utility function file
- `phonetic-word-embedding/`: phonetic word embedding related file
- `scripts/`: scripts for training and evaluation
- `simvecs/`: pre-trained vector file

## Installation
Before you begin, make sure to install the required dependencies :

### Env Installation

```bash
pip install -r requirements.txt 
```
The environment is as the following:
- `python` == 3.9.12
- `pip` == 24.3.1

## Execution
### Fine-tuning Llama for Lyric Generation 
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
