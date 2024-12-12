# Directory Designation
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))    # SLiYME/scripts/
project_root = os.path.abspath(os.path.join(current_dir, '..')) # SLiYME/
sys.path.append(project_root)   # Add project root to sys.path(SLiYME)

em_dir = os.path.join(project_root, 'phonetic-word-embedding', 'src')    # SLiYME/phonetic-word-embedding/src
sys.path.append(em_dir)

models_dir = os.path.join(project_root, 'models')
if models_dir not in sys.path:
    sys.path.append(models_dir)

utils_dir = os.path.join(project_root, 'utils')  # SLiYME/utils
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

data_dir = os.path.join(project_root, 'data')
if data_dir not in sys.path:
    sys.path.append(data_dir)
train_file_path = os.path.join(data_dir, 'train.json')
val_file_path = os.path.join(data_dir, 'val.json')

# Designate Specific Model You want to evaluate
trained_model_path = os.path.join(project_root, 'outputs', 'checkpoint-300')

import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from unsloth import FastLanguageModel
import torch
from tqdm import tqdm

max_seq_length = 2048
dtype = None
load_in_4bit = True

import embedding
from embedding import Dictionary
rhyme_dictionary = Dictionary("simvecs")  # 경로 수정

# Rhyme Score
def rhyme_score(previous_lines, generated_line, dictionary, num_words=1, position_weight_factor=1.0):
    """
    Compute Rhyme Score based on phonetic similarity.

    If a word is not in the dictionary, it contributes a similarity of 0.
    """
    def get_tail_words(line, num_words):
        words = line.split()
        return words[-num_words:] if len(words) >= num_words else words

    generated_tail_words = get_tail_words(generated_line, num_words)
    weights = []
    positional_weights = []
    num_lines = len(previous_lines)

    for idx, prev_line in enumerate(previous_lines):
        prev_tail_words = get_tail_words(prev_line, num_words)
        effective_num_words = min(len(prev_tail_words), len(generated_tail_words))

        positional_similarities = []
        for i in range(effective_num_words):
            try:
                similarity = dictionary.score(prev_tail_words[i], generated_tail_words[i])
                positional_similarities.append(similarity)
            except KeyError:
                # If the word is not in the dictionary, similarity is 0
                print(f"Word not found in dictionary: {prev_tail_words[i]} or {generated_tail_words[i]}")
                positional_similarities.append(0.0)

        if len(positional_similarities) > 0:
            avg_similarity = sum(positional_similarities) / len(positional_similarities)
        else:
            avg_similarity = 0

        weights.append(avg_similarity)
        position_weight = ((idx + 1) / num_lines) ** position_weight_factor
        positional_weights.append(position_weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    positional_weights = torch.tensor(positional_weights, dtype=torch.float32)
    combined_weights = weights * positional_weights
    if combined_weights.sum() > 0:
        combined_weights /= combined_weights.sum()

    rhyme_score_value = (weights * combined_weights).sum().item()
    return rhyme_score_value

# Metric 계산 함수
def compute_scores(predicted, reference, previous_lines):
    """
    Compute BERTScore, ROUGE, and Rhyme Score.
    """
    # BERTScore
    P, R, F1 = bert_score([predicted], [reference], model_type="bert-base-uncased", lang="en", verbose=False)

    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(predicted, reference)

    # Rhyme Score
    rhyme = rhyme_score(previous_lines, predicted, rhyme_dictionary, num_words=1, position_weight_factor=1.0)

    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item(),
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "rhyme_score": rhyme,
    }

# Evaluate Model
def evaluate_model(val_file_path, model, tokenizer):
    with open(val_file_path, "r", encoding="utf-8") as file:
        val_data = json.load(file)

    results = []
    total_scores = {
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "rhyme_score": 0.0,
    }
    num_samples = len(val_data)

    for entry in tqdm(val_data, desc="Evaluating", unit="sample"):
        song_name = entry["song_name"]
        context_lines = "\n".join([
            f"(Syllable Structure: {line['processed_line']}) {line['original_line']}"
            for line in entry["input_lines"]
        ])
        processed_line = f"(Syllable Structure: {entry['target_line']['processed_line']})"
        target_line = entry["target_line"]["original_line"]

        lyric_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Generate a song lyric line which will come next. You should follow Syllable structure in Next lyric line like Context.

### Input:
Context:
{context_lines}
Next lyric line:
{processed_line}

### Response:
"""
        inputs = tokenizer([lyric_prompt], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=128)
        generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated_response.split("### Response:")[-1].strip()

        previous_lines = [line["original_line"] for line in entry["input_lines"]]
        scores = compute_scores(response, target_line, previous_lines)

        for key in total_scores:
            total_scores[key] += scores[key]

        # Add previous_lines to the result for debugging or verification
        results.append({
            "song_name": song_name,
            "previous_lines": previous_lines,
            "generated_response": response,
            "target_line": target_line,
            "scores": scores
        })

    avg_scores = {key: value / num_samples for key, value in total_scores.items()}
    return results, avg_scores

# "checkpoint-300"

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=trained_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    val_file_path_specific = val_file_path
    results, avg_scores = evaluate_model(val_file_path_specific, model, tokenizer)

    output_file = "evaluation_results_with_rhyme.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump({"results": results, "average_scores": avg_scores}, file, indent=4, ensure_ascii=False)
