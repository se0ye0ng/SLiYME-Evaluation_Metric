import re

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
