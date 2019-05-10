from helpers.corpus_iterator import CorpusIterator
import pymorphy2
from wordfreq import zipf_frequency
import pandas as pd
import string
morph = pymorphy2.MorphAnalyzer()
# TODO: пунктуация тоже вошла в частоты - это не правильно
examples = []
lemmas_counts = {}
for filename in ['UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu',
                 'UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu']:
    print('Reading corpora {}...'.format(filename))
    with CorpusIterator(filename) as corpus_iter:
        for sent in corpus_iter:
            words = [tok.token.lower() for tok in sent]
            lemmas = [tok.lemma.lower() for tok in sent]
            gr_vals = [tok.grammar_value for tok in sent]
            for i, word in enumerate(words):

                if word in string.punctuation or word == '…':
                    continue

                word_freq = zipf_frequency(word, 'ru')
                lemma = lemmas[i]

                if lemma not in lemmas_counts:
                    lemmas_counts[lemma] = [word_freq, 1]
                else:
                    lemmas_counts[lemma][0] += word_freq
                    lemmas_counts[lemma][1] += 1

avg_frequencies = [
    (lemma, lemmas_counts[lemma][0] / lemmas_counts[lemma][1])
    if lemmas_counts[lemma][1] > 0 else 0
    for lemma in lemmas_counts
]

df = pd.DataFrame(avg_frequencies, columns=['lemma', 'frequency'])
print(df.head())
df.to_csv('data/lemma_frequencies.csv', encoding='utf8')
