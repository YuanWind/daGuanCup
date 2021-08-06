import pickle as pkl
from collections import Counter

import fire
from nltk import ngrams
from tqdm.auto import tqdm


def construct_ngram_dict(input_file="data/mlm_data/tokenizer_data.txt",
                         output_file="data/mlm_data/ngram_words.pkl",
                         min_frequence=10,
                         max_ngram=3):
    token_counter = Counter()
    for line in tqdm(open(input_file)):
        line = line.strip()
        for ngram_num in range(2, max_ngram + 1):
            token_counter.update(["".join(item) for item in ngrams(line, ngram_num)])
    ngram_dict = {}
    for k, v in tqdm(token_counter.most_common()):
        if v >= min_frequence:
            ngram_dict[k] = v
        else:
            break
    print(f'ngram_words: {len(ngram_dict)}')
    output_ngram_words = {}
    for k, v in sorted([(k, v) for k, v in ngram_dict.items()], key=lambda x: len(x[0])):
        sequence_len = len(k)
        for start_pos in range(0, sequence_len):
            for end_pos in range(start_pos+1, sequence_len):
                if k[start_pos:end_pos] in output_ngram_words and output_ngram_words[k[start_pos:end_pos]] == v and end_pos - start_pos > 1:
                    output_ngram_words.pop(k[start_pos:end_pos])
        output_ngram_words[k] = v
    print(f'ngram_words: {len(output_ngram_words)}')

    with open(output_file, 'wb') as fout:
        pkl.dump(output_ngram_words, fout)

if __name__ == '__main__':
    fire.Fire(construct_ngram_dict)
