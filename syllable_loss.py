import pronouncing 


def filter_phonemes_with_stress(word):
    phones = pronouncing.phones_for_word(word.lower())
    if phones:
        # 숫자가 포함된 음소만 남김
        stressed_phonemes = [phone for phone in phones[0].split() if phone[-1].isdigit()]
        syllable_list = ['[SYL]' for _ in stressed_phonemes]
        return syllable_list, stressed_phonemes  # 숫자가 있는 음소만 반환
    return ['[SYL]'], []  # 발음을 찾을 수 없는 경우 기본 음절 반환

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

def process_sentences_list(input_data):
    '''
    input_data(list) : sentence가 ['sentence', 'sentence'...] 형태로 묶인 리스트
    '''
    processed_sentences = [process_sentence(sentence) for sentence in input_data]
    return processed_sentences

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
            target.append("SYL")  # SYL로 매핑
        elif token.startswith('SEP]'):
            target.append("SEP")  # SEP로 매핑
    
    generated = []
    for token in generated_line.split('['):
        if token.startswith('SYL]'):
            generated.append("SYL")  # [SYL]은 SYL로 매핑
        elif token.startswith('SEP]'):
            generated.append("SEP")  # [SEP]은 SEP로 매핑

    # 2. 두 리스트의 길이를 맞추기 위해 패딩 (긴 쪽에 맞춤)
    max_len = max(len(target), len(generated))
    target += ["PAD"] * (max_len - len(target))
    generated += ["PAD"] * (max_len - len(generated))

    # 3. 로스 계산: 불일치율의 평균
    loss = 0
    for t, g in zip(target, generated):
        if t == g:
            continue  # 일치하는 경우 로스는 0
        else:
            loss += 1  # 불일치마다 1 추가

    return loss / max_len  # 불일치율의 평균 반환

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
    #print(f'Debug...target_syllable_count: {target_syllable_count}')
    #print(f'Debug...generated_syllable_count: {generated_syllable_count}')
    # 3. 음절 수 차이 계산 및 정규화
    loss = abs(target_syllable_count - generated_syllable_count) / target_syllable_count
    
    return loss

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

