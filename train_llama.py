import subprocess
import json
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from syllable_loss import filter_phonemes_with_stress
from transformers import BertTokenizer, BertModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset

# Install unsloth
subprocess.run(["pip", "install", "unsloth"], check=True)

# Uninstall unsloth and reinstall nightly version
subprocess.run(["pip", "uninstall", "unsloth", "-y"], check=True)
subprocess.run([
    "pip", "install", "--upgrade", "--no-cache-dir", "--no-deps",
    "git+https://github.com/unslothai/unsloth.git"
], check=True)


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Data Prep.
lyric_prompt = """Below is a description of a task where the model generates a song lyric. The input provides the context, including the original lines of the song and their corresponding syllable structures. Additionally, a processed line is provided to guide the syllable structure of the generated lyric. The model should generate a new lyric line that aligns with the context and adheres to the given syllable structure.

### Instruction:
Generate a single lyric line that fits the context provided by the input lines while matching the syllable structure of the given processed line.

### Input:
Context:
{context_lines}
Processed Line (Syllable Structure): {processed_line}

### Response:
{response}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


# JSON 파일 로드
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# Prompt 포맷 함수
def formatting_prompts_func(data):
    texts = []
    for entry in data:
        try:
            # 기본 정보 가져오기
            song_name = entry["song_name"]
            input_lines = entry["input_lines"]
            target_line = entry["target_line"]

            # input_lines 처리
            context_lines = "\n".join([
                f"- {line['original_line']} (Syllable Structure: {line['processed_line']})"
                for line in input_lines
            ])
            
            # target_line 처리
            processed_line = target_line["processed_line"]
            response = target_line.get("original_line", "")

            # 텍스트 생성
            text = lyric_prompt.format(
                song_name=song_name,
                context_lines=context_lines,
                processed_line=processed_line,
                response=response,
            ) + EOS_TOKEN
            texts.append(text)
        except KeyError as e:
            print(f"KeyError: {e} in entry {entry}")
        except Exception as e:
            print(f"Unexpected error: {e} in entry {entry}")
    return {"text": texts}


# JSON 파일 경로
file_path = "cleaned_data.json"  # JSON 파일 경로를 입력하세요.
data = load_json(file_path)

# Formatting Prompts
formatted_data = formatting_prompts_func(data)
dataset = Dataset.from_dict(formatted_data)


## Custom Loss
# -1. syllable_loss

def process_sentence(sentence):
    words = sentence.split()
    result = []

    for word in words:
        syllable_list, _ = filter_phonemes_with_stress(word)
        result.extend(syllable_list)
        result.append('[SEP]')  # 단어 사이에 [SEP] 추가

    # 마지막 [SEP] 제거
    if result and result[-1] == '[SEP]':
        result.pop()

    return "".join(result)

def calculate_syllable_sep_loss(gt_line, generated_line):
    '''
    음절 로스1. 개별 토큰 비교 
    - gt_line(str) : GT 음절 패턴. [SYL], [SEP]으로 구성된 문자열
    - generated_line(str) : 생성된 가사의 음절 패턴. [SYL], [SEP]으로 구성된 문자열
    Return:
    SYL, SEP의 불일치율 반영 로스 
    '''
    # 1. 음절과 SEP를 추출
    target = []
    for token in gt_line.split('['):
        if token.startswith('SYL]'):
            target.append(0)  # SYL을 0으로 매핑
        elif token.startswith('SEP]'):
            target.append(1)  # SEP를 1로 매핑
    
    generated = []
    for token in generated_line.split('['):
        if token.startswith('SYL]'):
            generated.append(0)  # SYL을 0으로 매핑
        elif token.startswith('SEP]'):
            generated.append(1)  # SEP를 1로 매핑

    # 2. 두 리스트의 길이를 맞추기 위해 패딩 (긴 쪽에 맞춤)
    max_len = max(len(target), len(generated))
    target += [2] * (max_len - len(target))  # 패딩을 2로
    generated += [2] * (max_len - len(generated))

    # 3. 로스 계산: 불일치율의 평균
    target_tensor = torch.tensor(target, dtype=torch.float32, requires_grad=False)
    generated_tensor = torch.tensor(generated, dtype=torch.float32, requires_grad=False)

    loss = (target_tensor != generated_tensor).float().mean()
    return loss

def calculate_syllable_count_loss(gt_line, generated_line):
    '''
    음절 로스2. 총 음절 개수 비교 
    - gt_line(str) : GT 음절 패턴. [SYL], [SEP]으로 구성된 문자열
    - generated_line(str) : 생성된 가사의 음절 패턴. [SYL], [SEP]으로 구성된 문자열

    Return:
      SYL 개수 차이 정규화 로스 

    '''
    # 1. gt_line에서 음절([SYL])의 개수 계산 
    target_syllable_count = gt_line.count("[SYL]")
    # 2. generated_line에서 음절([SYL])의 개수 계산
    generated_syllable_count = generated_line.count("[SYL]")
    # 3. 음절 수 차이 계산 및 정규화
    loss = abs(target_syllable_count - generated_syllable_count) / max(1, target_syllable_count)
    return torch.tensor(loss, requires_grad=True)  # Tensor로 변환

def syllable_loss(gt_line, generated_line):
    '''
    최종 음절 로스 : GT, Generated lines을 입력하면 전 처리 후 최종 로스 계산 
    - gt_line(str) : [SYL], [SEP]으로 구성된 음절 라벨 
    - generated_line(str) : 생성된 가사 
    '''
  
    processed_generated = process_sentence(generated_line)
    syllable_sep_loss = calculate_syllable_sep_loss(gt_line, processed_generated)
    syllable_count_loss = calculate_syllable_count_loss(gt_line, processed_generated)

    #print(f'Debug...syllable_sep:{syllable_sep_loss}, syllable_count:{syllable_count_loss}')
    # TODO : 각 Loss에 대한 coeff들 args로 받아서 넣을 수 있게 최종 구현 반영 필요 
    total_loss = syllable_sep_loss + syllable_count_loss
    return total_loss

# -2. Bert score Loss

# BERT model, tokenization 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# BERT Embedding 계산 함수
def get_bert_embeddings(sentence, tokenizer, model, device):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**tokens)  # no_grad 제거
    return outputs.last_hidden_state.mean(dim=1)  # 평균으로 문장 임베딩 생성

# bert score를 기반으로 한 loss 계산
def bert_score_loss(generated, gt, bert_model, bert_tokenizer, device="cuda"):
    """
    BERTScore Loss 계산
    """
    # BERT embedding
    generated_embeds = get_bert_embeddings(generated, bert_tokenizer, bert_model, device)
    gt_embeds = get_bert_embeddings(gt, bert_tokenizer, bert_model, device)

    # Cosine similarity
    cosine_sim = F.cosine_similarity(generated_embeds, gt_embeds, dim=1)
    loss = 1 - cosine_sim.mean()  # BERTScore를 최대화하도록
    return loss

# Custom Loss 정의
def custom_loss(gt_line, generated_line, bert_model, bert_tokenizer, device="cuda", alpha=0.5, beta=0.5):
    # 음절 손실
    processed_generated = process_sentence(generated_line)
    syllable_sep_loss = calculate_syllable_sep_loss(gt_line, processed_generated)
    syllable_count_loss = calculate_syllable_count_loss(gt_line, processed_generated)
    syllable_loss = syllable_sep_loss + syllable_count_loss

    # BERT Score 손실
    generated_embeds = get_bert_embeddings(generated_line, bert_tokenizer, bert_model, device)
    gt_embeds = get_bert_embeddings(gt_line, bert_tokenizer, bert_model, device)
    cosine_sim = F.cosine_similarity(generated_embeds, gt_embeds, dim=1)
    bert_loss = 1 - cosine_sim.mean()

    # Custom Loss 결합
    total_loss = alpha * syllable_loss + beta * bert_loss
    return total_loss

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom Loss를 SFTTrainer에 적용
        """
        
        # 모델 Forward pass
        # 모델 출력 설정
        model.config.return_dict = True
        model.config.output_hidden_states = True

        outputs = model(**inputs)
        logits = outputs.logits

        # logits 확인 및 계산
        if logits is None:
            if outputs.hidden_states is None:
                raise ValueError("Model does not return logits or hidden_states. Check the model configuration.")
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.to(model.lm_head.weight.dtype)
            logits = model.lm_head(hidden_states)
            outputs.logits = logits

        # labels 가져오기 및 범위 확인
        labels = inputs["labels"]
        vocab_size = self.tokenizer.vocab_size
        labels = torch.where(
            (labels >= 0) & (labels < vocab_size),
            labels,
            torch.tensor(self.tokenizer.pad_token_id).to(labels.device)
        )

        # 라벨과 생성된 텍스트 준비
        batch_size, seq_length = logits.size()[:2]
        predictions = logits.argmax(dim=-1)  # 가장 높은 확률의 토큰 선택
        generated_lines = [
            self.tokenizer.decode(predictions[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]
        gt_lines = []
        for i in range(batch_size):
            try:
                gt_lines.append(self.tokenizer.decode(labels[i], skip_special_tokens=True))
            except OverflowError as e:
                print(f"Error decoding label at index {i}: {labels[i]}")
                raise e

        # Custom Loss 계산
        total_loss = 0.0
        
        for gt_line, generated_line in zip(gt_lines, generated_lines):
            total_loss += custom_loss(
                gt_line=gt_line,
                generated_line=generated_line,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                device="cuda",
                alpha=0.7,
                beta=0.3
            )

        # 배치 평균 Loss 계산
        total_loss /= batch_size        
        return (total_loss, outputs) if return_outputs else total_loss
    

## Train the model

trainer = CustomSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # 짧은 시퀀스의 경우 속도 개선 가능
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")