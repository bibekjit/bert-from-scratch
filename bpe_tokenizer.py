from collections import Counter
import re
import numpy as np


class BPETokenizer:
    def __init__(self):
        self.special_toks = ['<pad>', '<mask>', '<unk>', '<cls>', '<sep>', '##']
        self.w2i = {'<pad>': 0, '<sep>': 2, '<mask>': 1, '<unk>': 3, '<cls>': 4, '##': 5}
        self.i2w = {self.w2i[k]: k for k in self.w2i}
        self.size = len(self.special_toks)
        self.tokens = {}
        self.words = {}
        self.vocab = {}
        self.oov_splits = {}

    def __call__(self, text, num_tok):
        corpus = ' '.join(text).split()
        tok_freq = Counter(corpus)
        tokens = sorted(corpus, key=lambda x: (tok_freq[x], x), reverse=True)

        count = 1

        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                count += 1
            else:
                self.tokens[tokens[i - 1]] = count

                if tokens[i - 1].isalpha():
                    self.words[tokens[i - 1]] = count

                count = 1

            if len(self.words) == num_tok:
                break

    def train(self, iterations=3, min_pair_freq=200):
        subwords = [w.replace('', '_')[1:-1] for w in self.words]

        for _ in range(iterations):
            combs = {}

            for i, w in enumerate(subwords):
                w = w.split('_')
                for j, _ in enumerate(w):
                    count = self.words[subwords[i].replace('_', '')]
                    comb = '_'.join(w[j:j + 2])
                    if comb in combs:
                        combs[comb] += count
                    else:
                        combs[comb] = count

            combs = {k: combs[k] for k in combs if combs[k] > min_pair_freq}

            for comb in combs:
                for i, sw in enumerate(subwords):
                    if comb in sw:
                        new_comb = comb.replace('_', '')
                        subwords[i] = sw.replace(comb, new_comb)

        for i, sw in enumerate(subwords):
            sw = sw.split('_')
            count = self.words[''.join(sw)]
            for x in sw:
                if x not in self.vocab:
                    self.vocab[x] = count
                else:
                    self.vocab[x] += count

        for w in self.tokens:
            if w not in self.words:
                if w not in self.vocab:
                    self.vocab[w] = 1
                else:
                    self.vocab[w] += 1

        for i, t in enumerate(self.vocab):
            self.w2i[t] = i + len(self.special_toks)
            self.i2w[i + len(self.special_toks)] = t

        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: len(x[0]), reverse=True))

    def _split_oov(self, word):

        if word in self.oov_splits:
            return self.oov_splits[word]

        elif word in self.w2i:
            return word

        subwords = {}
        for p in self.vocab:
            tmp = []
            if (p in word) and (p not in subwords):
                if word.index(p) == 0:
                    subwords[p] = []
                    sw = re.sub(p, '', word, 1)
                    for s in self.vocab:
                        if (s in sw) and (s not in subwords[p]):
                            subwords[p].append(s)
                else:
                    pass

        subw = []
        for p in subwords:
            rem = re.sub(p, '', word, 1)
            suff = subwords[p]
            suff = sorted(suff, key=lambda x: len(x))[::-1]
            sorter = [(sw, rem.index(sw)) for sw in suff]
            suff = sorted(sorter, key=lambda x: x[1])
            suff = [x[0] for x in suff]
            tmp = [p]
            for s in suff:
                if rem == '':
                    break

                elif (s[0] == rem[0]) and (s in rem):
                    rem = re.sub(s, '', rem)
                    tmp.append(s)

            if rem == '':
                subw.append(tmp)

        subw = sorted([x for x in subw], key=lambda x: len(x))

        if len(subw) > 0 and ''.join(subw[0]) == word:
            if word not in self.oov_splits:
                self.oov_splits[word] = ' ## '.join(subw[0])
            return ' ## '.join(subw[0])

        else:
            return '<unk>'

    def tokenize(self, seq):
        seq = np.asarray(seq.split())
        split_tok = np.vectorize(self._split_oov)
        seq = split_tok(seq)
        seq = ' '.join(seq).split()
        seq = [self.w2i[s] for s in seq]
        return seq

    @staticmethod
    def add_padding(seq, maxlen):
        padding = [0] * (maxlen - len(seq))
        seq = seq + padding
        return seq