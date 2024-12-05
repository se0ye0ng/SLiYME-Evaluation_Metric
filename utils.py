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
    context_match = re.search(r"### Input:\s*Context:\s*(.*?)\s*Requested Syllable Structure", prompt, re.DOTALL)
    if not context_match:
        print("--- Debugging Context Match ---")
        print("Pattern did not match!")
        print("Prompt:")
        print(prompt)
    context_text = context_match.group(1).strip() if context_match else None
    #print("---Context text---", context_text)
    processed_line_match = re.search(r"Requested Syllable Structure: (.*)", prompt)
    processed_line = processed_line_match.group(1).strip(" )") if processed_line_match else None
    response_match = re.search(r"### Expected Response:\n(.*)", prompt, re.DOTALL)
    response = response_match.group(1).strip() if response_match else None
    context_lines = []
    if context_text:
        context_lines = [
            re.sub(r"\(.*?\)", "", line).strip("- ").strip()  # 괄호 안 내용 제거
            for line in context_text.splitlines()
        ]
    #print("----context lines-----", context_lines)
    return context_lines, processed_line, response
def extract_output(prompt):
    output_match = re.search(r"Output:\n(.*)", prompt, re.DOTALL)
    if output_match:
        return output_match.group(1).strip()
    return None
