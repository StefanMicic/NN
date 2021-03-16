from keras import Input
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.layers.merge import add
from keras.models import Model

from data_generator import DataGenerator


class ImageCaptioningModel:
    def __init__(self, path_to_annotations):
        self.dataGenerator = DataGenerator(path_to_annotations)
        self.model = self.create_model()

    def create_model(self):
        embedding_dim = 200

        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation="relu")(fe1)

        inputs2 = Input(shape=(self.dataGenerator.max_length,))
        se1 = Embedding(
            self.dataGenerator.vocab_size, embedding_dim, mask_zero=True
        )(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation="relu")(decoder1)
        outputs = Dense(self.dataGenerator.vocab_size, activation="softmax")(
            decoder2
        )

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.summary()
        return model

    def train(self, epochs, batch_size):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        steps = len(self.dataGenerator.train_captions) // batch_size
        generator = self.dataGenerator(batch_size)
        self.model.fit(
            generator, epochs=epochs, steps_per_epoch=steps, verbose=1
        )
