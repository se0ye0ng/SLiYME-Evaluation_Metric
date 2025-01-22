import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
em_dir = os.path.join(project_root, 'phonetic-word-embedding', 'src')
if em_dir not in sys.path:
    sys.path.append(em_dir)

utils_dir = os.path.join(project_root, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

data_dir = os.path.join(project_root, 'data')
if data_dir not in sys.path:
    sys.path.append(data_dir)
train_file_path = os.path.join(data_dir, 'train.json')
val_file_path = os.path.join(data_dir, 'val.json')


import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import torch
from embedding import Dictionary
from utils import extract_context_processed_response

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

# Load rhyme dictionary
rhyme_dictionary = Dictionary("simvecs")

# Rhyme Score Function
def rhyme_score(previous_lines, generated_line, dictionary, num_words=1, position_weight_factor=1.0):
    def get_tail_words(line, num_words):
        words = line.split()
        return words[-num_words:] if len(words) >= num_words else words

    generated_tail_words = get_tail_words(generated_line, num_words)
    weights, positional_weights = [], []
    num_lines = len(previous_lines)

    for idx, prev_line in enumerate(previous_lines):
        prev_tail_words = get_tail_words(prev_line, num_words)
        positional_similarities = [
                dictionary.score(prev_tail_words[i], generated_tail_words[i])
                if i < len(prev_tail_words) and i < len(generated_tail_words)
                else 0
                for i in range(num_words)
        ]
        avg_similarity = sum(positional_similarities) / len(positional_similarities) if positional_similarities else 0
        weights.append(avg_similarity)
        positional_weights.append(((idx + 1) / num_lines) ** position_weight_factor)

    weights = torch.tensor(weights, dtype=torch.float32)
    positional_weights = torch.tensor(positional_weights, dtype=torch.float32)
    combined_weights = weights * positional_weights
    if combined_weights.sum() > 0:
        combined_weights /= combined_weights.sum()
    return (weights * combined_weights).sum().item()

# Compute all evaluation metrics
def compute_scores(predicted, reference, previous_lines):
    P, R, F1 = bert_score([predicted], [reference], model_type="bert-base-uncased", lang="en", verbose=False)
    print(f"BERT Score - Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(predicted, reference)
    print(f"ROUGE - Rouge1: {rouge_scores['rouge1'].fmeasure}, Rouge2: {rouge_scores['rouge2'].fmeasure}, RougeL: {rouge_scores['rougeL'].fmeasure}")
    rhyme = rhyme_score(previous_lines, predicted, rhyme_dictionary, num_words=1, position_weight_factor=1.0)
    print(f"Rhyme Score: {rhyme}")
    return {
            "bert_precision": P.mean().item(),
            "bert_recall": R.mean().item(),
            "bert_f1": F1.mean().item(),
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "rhyme_score": rhyme,
    }

def evaluate_model(val_file_path, model, tokenizer):
    """
    Load evaluation data, generate predictions, and compute evaluation metrics.
    """
    print("Loading validation data...")
    # Load validation data
    with open(val_file_path, "r", encoding="utf-8") as file:
        val_data = json.load(file)

    print(f"Validation data loaded. Number of samples: {len(val_data)}")

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

    # Process each sample in the validation data
    for entry in tqdm(val_data, desc="Evaluating", unit="sample"):
        song_name = entry["song_name"]
        context_lines = [line["original_line"] for line in entry["input_lines"]]
        target_line = entry["target_line"]["original_line"]

        # Generate prompt
        prompt = f"Context:\n" + "\n".join(context_lines) + "\nNext lyric line:\n"

        # Generate prediction using the model
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=128)
        generated_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Compute scores
        scores = compute_scores(generated_response, target_line, context_lines)
        print(f"Sample: {song_name}, Scores: {scores}")

        # Accumulate scores
        for key in total_scores:
            total_scores[key] += scores[key]

        # Append individual result
        results.append({
            "song_name": song_name,
            "generated_response": generated_response,
            "target_line": target_line,
            "scores": scores
        })

    # Compute average scores
    avg_scores = {key: value / num_samples for key, value in total_scores.items()}

    # Save results to JSON
    print("Saving results to JSON...")
    try:
        with open("evaluation_results.json", "w", encoding="utf-8") as file:
            json.dump({"results": results, "average_scores": avg_scores}, file, indent=4, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    return results, avg_scores

if __name__ == "__main__":
    # Load pre-trained model and tokenizer
#    model_name = "meta-llama/Llama-3.1-8B"  # Replace with your specific model path
#    model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")
#    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
#    You can replace the pre-trained model.
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define validation file path
    val_file_path = val_file_path

    # Run evaluation
    results, avg_scores = evaluate_model(val_file_path, model, tokenizer)
    print("Evaluation complete. Average Scores:")
    print(avg_scores)
