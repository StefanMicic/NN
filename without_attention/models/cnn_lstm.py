import os

import numpy as np
from keras import Input
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array


class CnnLSTM:
    def __init__(
        self,
        max_length,
        vocab_size,
        glove_path,
        glove: bool,
        wordtoix,
        EMBEDDING_DIM,
        MAX_SEQUENCE_LENGTH,
        num_words,
    ):

        embedding_dim = 200

        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation="relu")(fe1)

        inputs2 = Input(shape=(max_length,))
        if glove:
            se1 = self.create_glove_matrix(
                glove_path,
                vocab_size,
                wordtoix,
                EMBEDDING_DIM,
                MAX_SEQUENCE_LENGTH,
                num_words,
                inputs2,
            )
        else:
            se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation="relu")(decoder1)
        outputs = Dense(vocab_size, activation="softmax")(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.summary()

    def create_glove_matrix(
        self,
        glove_path,
        vocab_size,
        wordtoix,
        EMBEDDING_DIM,
        MAX_SEQUENCE_LENGTH,
        num_words,
        inputs2,
    ):
        embeddings_index = {}

        f = open(
            os.path.join(glove_path, "glove.6B.200d.txt"), encoding="utf-8"
        )
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        embeddings_index = {}
        embedding_dim = 200
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return Embedding(
            num_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False,
        )(inputs2)

    def data_generator(
        self,
        descriptions,
        photos,
        wordtoix,
        max_length,
        num_photos_per_batch,
        vocab_size,
    ):
        X1, X2, y = list(), list(), list()
        n = 0
        # loop for ever over images
        while 1:
            for key, desc_list in descriptions.items():
                n += 1
                photo = photos[key]
                for desc in desc_list:
                    seq = [
                        wordtoix[word]
                        for word in desc.split(" ")
                        if word in wordtoix
                    ]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        out_seq = to_categorical(
                            [out_seq], num_classes=vocab_size
                        )[0]
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)

                if n == num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0

    def train(
        self,
        train_descriptions,
        train_features,
        wordtoix,
        max_length,
    ):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        epochs = 30
        batch_size = 3
        steps = len(train_descriptions) // batch_size

        generator = self.data_generator(
            train_descriptions,
            train_features,
            wordtoix,
            max_length,
            batch_size,
        )
        self.model.fit(
            generator, epochs=epochs, steps_per_epoch=steps, verbose=1
        )
