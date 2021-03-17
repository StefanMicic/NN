from gensim.models import Word2Vec


class Word2VecModel:
    def __init__(self, data):
        self.data = data

    def __call__(self):
        model = Word2Vec(self.data, size=200)
        model.save("word2vec.model")
