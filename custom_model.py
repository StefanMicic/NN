import efficientnet.keras as enet
from keras import Input
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalMaxPooling2D,
)
from keras.layers.merge import add
from keras.models import Model, Sequential

from custom_model_data_generator import CustomDataGenerator


class CustomModel:
    def __init__(self, path_to_annotations):
        self.dataGenerator = CustomDataGenerator(path_to_annotations)
        self.model = self.create_model()

    def create_model(self):
        conv_base = enet.EfficientNetB2(
            weights="imagenet",
            include_top=False,
            input_shape=(126, 126, 3),
        )
        model = Sequential()
        model.add(conv_base)
        model.add(GlobalMaxPooling2D(name="gap"))
        model.add(Flatten(name="flatten"))
        model.add(Dropout(0.2, name="dropout_out"))

        model.add(Dense(256, activation="relu", name="fc1"))

        embedding_dim = 200

        inputs1 = Input(shape=(126, 126, 3))
        fe1 = model(inputs1)

        inputs2 = Input(shape=(22,))
        se1 = Embedding(32, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)

        se3 = LSTM(256)(se2)

        decoder1 = add([fe1, se3])
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
