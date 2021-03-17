import os
from typing import Dict

import numpy as np
from keras import Input
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from attention import Attention
from data_generator import DataGenerator


class ImageCaptioningModel:
    """Model with pretrained feature extractor."""

    def __init__(
        self,
        path_to_annotations,
        glove_path: str,
        glove: bool = False,
        attention: bool = False,
    ):
        self.dataGenerator = DataGenerator(path_to_annotations)
        self.model = self.create_model(attention, glove)
        self.glove_path = glove_path

    def create_glove_embeding_matrix(self) -> Dict:
        """Creates matrix if you want to use pretrained embedding layer."""
        embeddings_index = {}
        f = open(
            os.path.join(self.glove_path, "glove.6B.200d.txt"),
            encoding="utf-8",
        )
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        embedding_dim = 200
        embedding_matrix = np.zeros(
            (self.dataGenerator.vocab_size, embedding_dim)
        )
        for word, i in self.dataGenerator.wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def create_model(
        self, attention: bool = False, glove: bool = False
    ) -> Model:
        """ Creates model's architecture."""
        embedding_dim = 200

        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation="relu")(fe1)

        inputs2 = Input(shape=(self.dataGenerator.max_length,))
        if glove:
            se1 = Embedding(
                self.dataGenerator.vocab_size,
                embedding_dim,
                weights=[self.create_glove_embeding_matrix()],
                trainable=False,
            )(inputs2)
        else:
            se1 = Embedding(
                self.dataGenerator.vocab_size, embedding_dim, mask_zero=True
            )(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        if attention:
            se3 = Attention()(se3)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation="relu")(decoder1)
        outputs = Dense(self.dataGenerator.vocab_size, activation="softmax")(
            decoder2
        )

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.summary()
        return model

    def train(self, epochs, batch_size) -> None:
        """Compiles and trains a model."""
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        steps = len(self.dataGenerator.train_captions) // batch_size
        generator = self.dataGenerator(batch_size)
        self.model.fit(
            generator, epochs=epochs, steps_per_epoch=steps, verbose=1
        )

    def predict(self, image_path) -> str:
        """Load an image from specified path and predicts caption."""
        in_text = "<start>"
        for i in range(self.dataGenerator.max_length):
            sequence = [
                self.dataGenerator.wordtoix[w]
                for w in in_text.split()
                if w in self.dataGenerator.wordtoix
            ]
            sequence = pad_sequences(
                [sequence], maxlen=self.dataGenerator.max_length
            )
            prediction = self.model.predict(
                [np.asarray([self.dataGenerator.encode(image_path)]), sequence]
            )
            word = self.dataGenerator.ixtoword[np.argmax(prediction)]
            in_text += " " + word
            if word == "<end>":
                break

        return in_text
