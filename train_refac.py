import subprocess
import json
import torch
import re
import argparse
import torch.nn.functional as F
from unsloth import FastLanguageModel
from loss import SyllableLoss, BERTLoss, RhymeLoss
from transformers import BertTokenizer, BertModel, GenerationConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, 'phonetic-word-embedding', 'src')
sys.path.append(target_dir)
import embedding
from embedding import Dictionary
from utils import extract_context_processed_response, extract_output

# Below is only for first attempts
# Install unsloth
#subprocess.run(["pip", "install", "unsloth"], check=True)

# Uninstall unsloth and reinstall nightly version
#subprocess.run(["pip", "uninstall", "unsloth", "-y"], check=True)
#subprocess.run([
#    "pip", "install", "--upgrade", "--no-cache-dir", "--no-deps",
#    "git+https://github.com/unslothai/unsloth.git"
#], check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with custom loss functions.")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B",
                        help="Name of the pre-trained model.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model.")
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", None],
                        help="Data type for the model (e.g., float16, bfloat16).")
    parser.add_argument("--load_in_4bit", action="store_true", default = True,
                        help="Enable 4-bit quantization to save memory.")

    # Dataset and file paths
    parser.add_argument("--train_file", type=str, default="cleaned_data.json",
                        help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model checkpoints and outputs.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training.")
    parser.add_argument("--max_steps", type=int, default=60,
                        help="Maximum number of training steps.")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Steps interval for logging.")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for reproducibility.")

    # Loss weights
    parser.add_argument("--w_syllable", type=float, default=1.0,
                        help="Weight for the syllable loss.")
    parser.add_argument("--w_bert", type=float, default=1.0,
                        help="Weight for the BERT loss.")
    parser.add_argument("--w_rhyme", type=float, default=1.0,
                        help="Weight for the rhyme loss.")

    # Generation configuration (optional)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling for text generation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling for text generation.")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated sequences.")

    return parser.parse_args()


def load_and_prepare_data(train_file, tokenizer):
    with open(train_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    def formatting_prompts_func(data, EOS_token):

        texts = []
        responses = []

        lyric_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        Generate a song lyric line which will come after Context. You should follow Syllable structure in Next lyric line like Context.

        ### Input:
        Context:
        {context_lines}
        Next lyric line:
        (Syllable Structure : {response_structure})

        ### Response:
        {response}
        """

        for entry in data:
            try:
                input_lines = entry["input_lines"]
                target_line = entry["target_line"]

                # Context 생성
                context_lines = "\n".join(
                    [f"(Syllable Structure: {line['processed_line']}) {line['original_line']}" for line in input_lines]
                )

                # Response 데이터
                response = target_line["original_line"]
                response_structure = target_line["processed_line"]

                # Text (Prompt) 생성
                text = lyric_prompt.format(
                    context_lines=context_lines,
                    response_structure=response_structure,
                    response=response
                ) + EOS_token

                texts.append(text)      # Prompt 저장
                responses.append(response)  # Response 저장

            except KeyError as e:
                print(f"KeyError: {e}")

        return {"text": texts}

    EOS_token = tokenizer.eos_token
    formatted_data = formatting_prompts_func(data, EOS_token)
    print('==============DATA DEBUG==============')
    from IPython import embed; embed(colors="neutral")  # XXX DEBUG  # yapf: disable
    return Dataset.from_dict(formatted_data)


def initialize_trainer(args, model, tokenizer, dataset):
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True if args.dtype == "float16" else False,
        bf16=True if args.dtype == "bfloat16" else False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        custom_args=args
    )
    return trainer

def custom_loss(gt_prompt, generated_output,w_syllable, w_bert, w_rhyme):
    p_dict = Dictionary("simvecs")

    context_lines, processed_line, response = extract_context_processed_response(gt_prompt)
    generated_line = extract_output(generated_output)

    print('Debugging...')
    print(f'Requsted syllable : {processed_line}')
    print(f'GT lyric : {response}')
    print(f'Generated line : {generated_line}')
    print('<<Full output of LLama>>')
    print(generated_output)

    syllable_loss = SyllableLoss(coeff_sep = 1.0, coeff_count = 1.0)
    bert_loss = BERTLoss(model_name="bert-base-uncased")
    rhyme_loss = RhymeLoss(dictionary=p_dict, num_words=1, position_weight_factor=1.0)

    l_syllable = syllable_loss(processed_line, generated_line)
    l_lyric_bert = bert_loss(response, generated_line)
    l_rhyme = rhyme_loss(context_lines, generated_line)

    #l_prompt_bert = bert_loss(generated_output,


    #Debug
    print(f'Syllalbe Loss : {l_syllable}, Bert Loss : {l_lyric_bert}, Rhyme Loss : {l_rhyme}')

    # Custom Loss 결합
    total_loss = w_syllable * l_syllable + w_bert * l_lyric_bert + w_rhyme * l_rhyme
    return total_loss

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, custom_args=None, **kwargs):
        """
        Custom Trainer initialization to accept additional arguments.
        """
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom Loss를 SFTTrainer에 적용
        - inputs : Dictionary of input_ids, attention mask, labels
        """

        model.config.return_dict = True
        model.config.output_hidden_states = True
        print('==============DEBUG : Loss============')
        from IPython import embed; embed(colors="neutral")  # XXX DEBUG  # yapf: disable
        original_loss = super(SFTTrainer,self).compute_loss(model,inputs, return_outputs=False)

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
        #vocab_size = self.tokenizer.vocab_size #deprecated
        vocab_size = self.processing_class.vocab_size
        labels = torch.where(
            (labels >= 0) & (labels < vocab_size),
            labels,
            torch.tensor(self.processing_class.pad_token_id).to(labels.device)
        )

        # 라벨과 생성된 텍스트 준비
        batch_size, seq_length = logits.size()[:2]
        predictions = logits.argmax(dim=-1)  # 가장 높은 확률의 토큰 선택
        generated_output = [
            self.processing_class.decode(predictions[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]
        gt_prompt = []
        for i in range(batch_size):
            try:
                gt_prompt.append(self.processing_class.decode(labels[i], skip_special_tokens=True))
            except OverflowError as e:
                print(f"Error decoding label at index {i}: {labels[i]}")
                raise e
        # Custom Loss 계산
        total_loss = 0.0

        for gt_p, generated_p in zip(gt_prompt, generated_output):

            total_loss += custom_loss(
                gt_prompt=gt_p,
                generated_output=generated_p,
                w_syllable=self.custom_args.w_syllable,
                w_bert=self.custom_args.w_bert,
                w_rhyme = self.custom_args.w_rhyme
            )
        # 배치 평균 Loss 계산
        total_loss /= batch_size
        total_loss = total_loss + original_loss
        return (total_loss, outputs) if return_outputs else total_loss

def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = args.dtype,
        load_in_4bit = args.load_in_4bit,
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
    dataset = load_and_prepare_data(args.train_file, tokenizer)

    response_template = "### Response:"
    instruction_template = "### Intruction:"


    collator = DataCollatorForCompletionOnlyLM(
        #instruction_template=instruction_template,
        response_template = "### Response:",
        tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True if args.dtype == "float16" else False,
        bf16=True if args.dtype == "bfloat16" else False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        custom_args=args,
        data_collator=collator
    )
    print('Starting training')
    trainer.train()

    print(f"Saving model and tokenizer")
    model.save_pretrained("lora_model")  # 모델 저장
    tokenizer.save_pretrained("lora_model")

if __name__ == "__main__":
    main()
