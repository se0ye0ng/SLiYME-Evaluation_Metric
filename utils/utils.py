import re
import pronouncing

def extract_context_processed_response(prompt):
    context_match = re.search(r"### Input:\s*Context:\s*(.*?)\s*Next lyric line:\s", prompt, re.DOTALL)
    processed_line_match = re.search(r"Next lyric line:\s*\(Syllable Structure\s*:\s*([\[\]SYLSEP]+)\)", prompt)
    response_match = re.search(r"### Response:\n(.*)", prompt, re.DOTALL)

    context_lines = [
            re.sub(r"\(Syllable Structure:.*?\)", "", line).strip()
            for line in context_match.group(1).splitlines()
    ] if context_match else []
    processed_line = processed_line_match.group(1).strip(" )") if processed_line_match else None
    response = response_match.group(1).strip() if response_match else None
    return context_lines, processed_line, response

def calculate_rhyme_similarity(word1, word2):
        phonemes1 = pronouncing.phones_for_word(word1.lower())
        phonemes2 = pronouncing.phones_for_word(word2.lower())
        if not phonemes1 or not phonemes2:
                return 0.0
        last_phoneme1 = phonemes1[0].split()[-1]
        last_phoneme2 = phonemes2[0].split()[-1]
        return 1.0 if last_phoneme1 == last_phoneme2 else 0.0