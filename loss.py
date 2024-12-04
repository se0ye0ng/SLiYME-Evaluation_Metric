import pronouncing
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import re

def filter_phonemes_with_stress(word):
    phones = pronouncing.phones_for_word(word.lower())
    if phones:
        stressed_phonemes = [phone for phone in phones[0].split() if phone[-1].isdigit()]
        syllable_list = ['[SYL]' for _ in stressed_phonemes]
        return syllable_list, stressed_phonemes
    return ['[SYL]'], []

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

class SyllableLoss:
    def __init__(self, coeff_sep=1.0, coeff_count=1.0):
        self.coeff_sep = coeff_sep
        self.coeff_count = coeff_count

    def calculate_syllable_sep_loss(self, gt_line, generated_line):
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

    def calculate_syllable_count_loss(self, gt_line, generated_line):
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

    def __call__(self, gt_line, generated_line):
        processed_generated = process_sentence(generated_line)
        sep_loss = self.calculate_syllable_sep_loss(gt_line, processed_generated)
        count_loss = self.calculate_syllable_count_loss(gt_line, processed_generated)
        total_loss = self.coeff_sep * sep_loss + self.coeff_count * count_loss
        return total_loss

class BERTLoss:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def get_embeddings(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)

    def __call__(self, gt, generated):
        generated_embeds = self.get_embeddings(generated)
        gt_embeds = self.get_embeddings(gt)
        cosine_sim = F.cosine_similarity(generated_embeds, gt_embeds, dim=1)
        cosine_sim = cosine_sim.clamp(min=0, max=1)
        return 1 - cosine_sim.mean()

class RhymeLoss:
    def __init__(self, dictionary, num_words=1, position_weight_factor=1.0):
        """
        Initialize RhymeLoss with parameters.

        Parameters:
        - dictionary: Pre-loaded Dictionary object for phonetic similarity.
        - num_words (int): Number of tail words to consider for similarity calculation.
        - position_weight_factor (float): Additional weight for lines closer to the generated line.
        """
        self.dictionary = dictionary
        self.num_words = num_words
        self.position_weight_factor = position_weight_factor

    def _get_tail_words(self, line, num_words):
        """
        Helper method to safely get the tail words of a line.
        """
        # 1. Remove leading/trailing whitespaces
        line = line.strip()

        # 2. Match lines starting with '-' or a word
        if not re.match(r'^(-|\w+)', line):
            print("Empty starts due to llama outputs")
            return ""  # Return empty if the line doesn't start with '-' or a word

        # 3. Truncate at the first occurrence of 3 or more consecutive special characters
        words = re.split(r'[^\w\s]{3,}', line)[0]
        words = re.sub(r"[^\w\s'.]", "", words)
        words = words.split()
        if len(words) < num_words:
            return words  # Use all available words if fewer than `num_words`
        return words[-num_words:]

    def __call__(self, previous_lines, generated_line):
        """
        Calculate rhyme loss between previous lines and the generated line.

        Parameters:
        - previous_lines (list of str): List of previous lyric lines.
        - generated_line (str): Generated lyric line.

        Returns:
        - torch.Tensor: Calculated loss value.
        """
        # Extract tail words from the generated line
        generated_tail_words = self._get_tail_words(generated_line, self.num_words)
        print(f'[Debug] tail of generated output : {generated_tail_words}')

        # Initialize storage for weights and losses
        weights = []
        positional_weights = []
        all_positional_losses = []
        num_lines = len(previous_lines)

        for idx, prev_line in enumerate(previous_lines):
            # Extract tail words from the previous line
            prev_tail_words = self._get_tail_words(prev_line, self.num_words)
            # Determine the actual number of words to compare
            effective_num_words = min(len(prev_tail_words), len(generated_tail_words))

            # Compute positional losses
            positional_losses = []
            for i in range(effective_num_words):
                similarity = self.dictionary.score(prev_tail_words[i], generated_tail_words[i])
                loss = 1 - similarity  # Maximize similarity
                positional_losses.append(loss)
            all_positional_losses.append(positional_losses)

            # Compute average similarity
            if len(positional_losses) > 0:
                avg_similarity = sum([1 - loss for loss in positional_losses]) / len(positional_losses)
            else:
                avg_similarity = 0  # No comparison possible, fallback to 0 similarity
            weights.append(avg_similarity)

            # Compute positional weight
            position_weight = (idx + 1) / num_lines
            position_weight = position_weight ** self.position_weight_factor
            positional_weights.append(position_weight)

        # Combine similarity weights with positional weights
        weights = torch.tensor(weights, dtype=torch.float32)
        positional_weights = torch.tensor(positional_weights, dtype=torch.float32)
        combined_weights = weights * positional_weights
        combined_weights /= combined_weights.sum()

        # Calculate total loss
        total_loss = 0.0
        for positional_losses, weight in zip(all_positional_losses, combined_weights):
            for loss in positional_losses:
                total_loss += weight * loss

        return torch.tensor(total_loss, dtype=torch.float32)
