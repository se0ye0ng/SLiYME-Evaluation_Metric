"""
Embedding
==========

Loads the vitz-1973-experiment dataset and generate embedding scores for them.
"""

import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

embedding_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(embedding_dir, '..', '..'))
exec_dir = os.path.join(project_root, 'scripts')
if exec_dir not in sys.path:
    sys.path.append(exec_dir)
utils_dir = os.path.join(project_root, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
    
from utils import calculate_rhyme_similarity

class Dictionary:
    def __init__(self, filepath, encoding="latin1"):
        self.words = list()
        self.lookup = dict()
        dictionary = list()

        print("loading...", file=sys.stderr)
        for i, line in enumerate(open(filepath, encoding=encoding)):
            line = line.strip()
            word, vec_s = line.split("  ")
            vec = [float(n) for n in vec_s.split()]
            self.lookup[word] = i
            dictionary.append(vec)
            self.words.append(word)
        print(f'Total words: {len(self.words)}', file=sys.stderr)
        self.dictionary = np.array(dictionary)
        self.norms = normalize(self.dictionary, axis=1)
        print('min Norm', np.min(self.norms))
        print('max Norm', np.max(self.norms))

    def vec(self, word):
        return self.dictionary[self.lookup[word.strip().upper()], :]

    def score(self, word1, word2):
        try:
            v1 = self.norms[self.lookup[word1.strip().upper()], :]
            v2 = self.norms[self.lookup[word2.strip().upper()], :]
            return np.sum(v1*v2)
        except KeyError:
            # simvecs 사전에 없는 단어들의 경우 phonetic similarity(발음 기반 유사도) 활용
            print(f"KeyError: Missing {word1} or {word2} in dictionary. Calculating phonetic similarity.")
            similarity = calculate_rhyme_similarity(word1, word2)
        
            # 새로운 단어를 딕셔너리에 추가
            word1_upper = word1.strip().upper()
            self.lookup[word1_upper] = len(self.words)
            self.words.append(word1_upper)
            
            # 임의의 벡터로 확장 (유사도를 기반으로 벡터 업데이트 가능)
            random_vector = np.random.rand(self.dictionary.shape[1]) * similarity
            self.dictionary = np.vstack([self.dictionary, random_vector])
            self.norms = normalize(self.dictionary, axis=1)

            return similarity
        
    def word(self, vec, n=None):
        v = vec / np.linalg.norm(vec)
        dots = np.dot(self.norms, v)
        if n is None:
            return self.words[np.argmax(dots)]
        return [(self.words[x], dots[x]) for x in np.argsort(-dots)[:n]]
        # return [self.words[x] for x in np.argsort(-dots)[:n]]


def compare(word, dictionary, res_dir):
    filepath = os.path.join(res_dir, f'vitz-1973-experiment-{word}.csv')
    df = pd.read_csv(filepath)
    df['score'] = df.apply(
        lambda row: dictionary.score(row['word'], word), axis=1)
    df['actual'] = word
    return df[['actual', 'word', 'score']]

def main(args):
    dictionary = Dictionary(args.input, encoding=args.encoding)
    words = ['sit', 'plant', 'wonder', 'relation']
    df =  pd.concat([compare(w, dictionary, args.res) for w in words], ignore_index=True)
    df.to_csv(args.output, index=False)


def _get_args():
    parser = ArgumentParser(
        os.path.basename(__file__), description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("input", type=str, help='embedding file path')
    parser.add_argument("output", type=str, help='output score file path')
    parser.add_argument("-e", "--encoding", default='latin1', type=str, help='File encoding (default: latin1)')
    res_dir = os.path.join(os.path.dirname(__file__), '..', 'res')
    parser.add_argument("-r", "--res", default=res_dir, type=str, help=f'Resourse directory (default: {res_dir})')
    return parser.parse_args()

if __name__ == '__main__':
    main(_get_args())
