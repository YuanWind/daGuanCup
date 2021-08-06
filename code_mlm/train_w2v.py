import fire
from gensim.models import Word2Vec
import random
from tqdm.auto import tqdm
import pickle as pkl
from gensim.models.callbacks import CallbackAny2Vec
random.seed(9527)

class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


ngram_words = pkl.load(open("data/mlm_data/ngram_words.pkl",
                            "rb"))


def get_tokenized_sentence(sentence,
                           ngram_words,
                           max_ngram=4):
    output_tokens = []
    sentence_len = len(sentence)
    i = 0
    while i < sentence_len:
        words = [sentence[i]]
        for ngram_num in range(2, max_ngram + 1):
            if sentence[i:i + ngram_num] in ngram_words:
                words.append(sentence[i:i + ngram_num])
        token = random.choice(words)
        output_tokens.append(token)
        i += len(token)
    return " ".join(output_tokens)


def main(token_data_file="data/mlm_data/tokenizer_data.txt",
         out_file="data/mlm_data/w2v.model"
         ):
    sentences = []

    for line in tqdm(open(token_data_file)):
        line = line.strip()
        cur_sentences = []
        for i in range(10):
            output_sentence = get_tokenized_sentence(line, ngram_words)
            if output_sentence not in cur_sentences:
                cur_sentences.append(output_sentence)
        sentences.extend(cur_sentences)

    sentences = [sentence.split() for sentence in sentences]
    random.shuffle(sentences)

    model = Word2Vec(sentences=sentences,
                     vector_size=128,
                     window=5,
                     compute_loss=True,
                     min_count=1,
                     seed=9527,
                     workers=32,
                     alpha=0.5,
                     min_alpha=0.0005,
                     epochs=100,
                     batch_words=int(4e4),
                     callbacks=[callback()])

    model.wv.save(out_file)


if __name__ == '__main__':
    fire.Fire(main)
