import tensorflow as tf

from models.attention_decoder import RNN_Decoder
from models.encoder import CNN_Encoder


class TrainModel:
    def __init__(
        self, tokenizer, embedding_dim=256, units=512, vocab_size=5001
    ):
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam()

    def loss_function(self, real, pred):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
        )
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(
                    dec_input, features, hidden
                )
                loss += self.loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])
        trainable_variables = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def train(self, dataset, num_steps, EPOCHS=20, start_epoch=0):
        loss_plot = []
        for epoch in range(start_epoch, EPOCHS):
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss
            loss_plot.append(total_loss / num_steps)
            print(f"Epoch {epoch}: Loss = {total_loss / num_steps}")
