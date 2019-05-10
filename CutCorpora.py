from helpers.example import Example
from helpers.field import Field
from helpers.corpus_iterator import CorpusIterator
import pandas as pd


def get_lemma_frequencies(csvfile, lemmas_to_drop):
    print('Read frequencies...')
    print()
    freqs = pd.read_csv(csvfile, index_col=0)
    freqs = freqs.sort_values(by=['frequency'], ascending=False)
    print('Most frequent lemmas:')
    print(freqs.head())
    print()
    print('Least frequent lemmas:')
    print(freqs.tail())
    print()

    if type(lemmas_to_drop) == float:
        lemmas_to_drop = int(lemmas_to_drop * freqs.shape[0])
    treshold_freq = freqs['frequency'].values[-lemmas_to_drop]
    print('Cutting out lemmas with frequency >= {}'.format(treshold_freq))
    print()
    cutted = freqs.loc[freqs['frequency'] < treshold_freq]
    print('Most frequent lemmas after cut:')
    print(cutted.head())
    print()

    return cutted


def read_corpora(corpora_filename_conllu, freqs):
    examples = {'target_idx': [], 'sentence': [], 'lemmas': [], 'gram_vals': []}
    print('Reading corpora {}...'.format(corpora_filename_conllu))
    with CorpusIterator(corpora_filename_conllu) as corpus_iter:
        for sent in corpus_iter:
            words = [tok.token for tok in sent]  # lower()
            lemmas = [tok.lemma.lower() for tok in sent]
            gram_vals = [tok.grammar_value for tok in sent]

            for i, lemma in enumerate(lemmas):
                if lemma in freqs['lemma'].values:
                    examples['target_idx'].append(i)
                    examples['sentence'].append(words)
                    examples['lemmas'].append(lemmas)
                    examples['gram_vals'].append(gram_vals)
    return pd.DataFrame(examples)

#
# def export_to_csv(examples):
#     df = pd.DataFrame(examples)
#     df.to_csv('data/syntagrus_low_frequent_lemmas.csv', encoding='utf8')


if __name__ == '__main__':
    freqs = get_lemma_frequencies('data/lemma_frequencies.csv', 0.45)
    train_examples = read_corpora('UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu', freqs)
    dev_examples = read_corpora('UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu', freqs)
    final_examples = pd.concat([train_examples, dev_examples])
    final_examples.to_csv('data/syntagrus_low_frequent_lemmas.csv', encoding='utf8')