## S-LiYME: Scores for Lyrics in Rhyme
Official Repository for SLiYME Project @DL&amp;NLP
\rightarrow
# Evaluation Metric for Lyric Generation
---
## Introduction
**S-LiYME; Evaluation Metric System**
S-LiYME generates song lyrics by considering syllable structure, context consistency, and rhyme.
*This Repository is 'evaluation metric system' for lyric generation model including S-LiYME.*

## Overview
This system evaluates the quality of generated lyrics using:
- **BERT Score**: Evaluates semantic similarity.
- **ROUGE Score**: Evaluates n-gram overlap.
- **Rhyme Score**: Evaluates phonetic similarity between lines.

## Structure
- `evaluate_metric.py`: Main script for evaluation.
- `utils.py`: Utility functions.
- `simvecs`: **Example** Phonetic similarity dictionary. Pre-trained vector file (with `llama-3.1-8b').
- `data/`: dataset for training and evaluation
- `phonetic-word-embedding/`: phonetic word embedding related file

## Setup
1. Install environment:
```bash
conda env create -f environment.yaml
conda activate eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare the data
place your evaluation dataset(ex.`val.json`) in the `data/` directory.

2. Run the Evaluation
To evaluate a model:
```bash
python scripts/evaluate_metric.py
```

3. View the Results
The results will be saved as a JSON file (`evaluation_results.json`) in the root directory.

---
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
SLiYME-Evaluation_Metric/
├── data/
│   ├── train.json          # 훈련 데이터셋
│   ├── val.json            # 평가 데이터셋
├── scripts/
│   └── evaluate_metric.py  # 평가 메트릭 메인 코드
├── utils/
│   └── utils.py            # 유틸리티 함수
├── phonetic-word-embedding/
├── simvecs                # 음운 유사도 계산을 위한 사전 파일
├── requirements.txt       # 필요한 라이브러리 목록
├── environment.yaml       # conda environment 설치 파일
└── README.md              # 프로그램 설명 및 실행 가이드
```
