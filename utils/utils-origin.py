import re

def extract_context_processed_response(prompt):
    '''
    Extract previous lyrics, requested syllable and gt lyric from prompt.
    Input:
        - prompt : prompt from input
    Output:
        - context_lines(list) : previous lines of targeted line
        - processed_line(str) : Regquested syllable structure
        - response(str) : GT Lyric of processed line
    '''
    context_match = re.search(r"### Input:\s*Context:\s*(.*?)\s*Next lyric line:\s", prompt, re.DOTALL)
    if not context_match:
        print("--- Debugging Context Match ---")
        print("Pattern did not match!")
        print("Prompt:")
        print(prompt)
    context_text = context_match.group(1).strip() if context_match else None
    processed_line_match = re.search(r"Next lyric line:\s*\(Syllable Structure\s*:\s*([\[\]SYLSEP]+)\)", prompt)
    processed_line = processed_line_match.group(1).strip(" )") if processed_line_match else None
    response_match = re.search(r"### Response:\n(.*)", prompt, re.DOTALL)
    response = response_match.group(1).strip() if response_match else None
    context_lines = []
    if context_text:
        context_lines = [
            re.sub(r"\(Syllable Structure:.*?\)", "", line).strip()
            for line in context_text.splitlines()
        ]
    return context_lines, processed_line, response

def extract_output(prompt):
    output_match = re.search(r"### Response:\n(.*)", prompt, re.DOTALL)
    if output_match:
        return output_match.group(1).strip()
    return None
