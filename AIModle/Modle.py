import numpy as np
import Data.LoadData as LD
class AImodle:
    def __init__(self):
        input_size = 28 * 28
        hidden_size = 128

        try:
            model_data = np.load("model_weights.npz")
            print("Gespeichertes Modell gefunden, lade Gewichte...")

            self.inputwights = model_data['inputwights']
            self.hiddenlayersBias = model_data['hiddenlayersBias']
            self.wights = model_data['wights']
            self.outputwights = model_data['outputwights']
            self.outputBias = model_data['outputBias']

            self.hiddenlayers = [np.zeros(hidden_size, dtype=np.float32) for _ in range(len(self.hiddenlayersBias))]
            self.outputLayer = np.zeros(10, dtype=np.float32)

        except (FileNotFoundError, IOError):
            print("Kein gespeichertes Modell gefunden, initialisiere mit Standardwerten...")

            # Standard-Initialisierung mit Xavier/He-Methode
            self.inputwights = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
            self.hiddenlayers = [np.zeros(hidden_size, dtype=np.float32) for _ in range(2)]
            self.hiddenlayersBias = [np.zeros(hidden_size, dtype=np.float32) for _ in range(2)]
            self.wights = [np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)]

            self.outputLayer = np.zeros(10, dtype=np.float32)
            self.outputwights = np.random.randn(hidden_size, 10).astype(np.float32) * np.sqrt(2.0 / hidden_size)
            self.outputBias = np.zeros(10, dtype=np.float32)

    def ForwardPass(self, x, return_intermediates=False):
        z1 = np.dot(x, self.inputwights) + self.hiddenlayersBias[0]
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.wights[0]) + self.hiddenlayersBias[1]
        a2 = self.sigmoid(z2)

        z_output = np.dot(a2, self.outputwights) + self.outputBias
        output = self.sigmoid(z_output)

        if return_intermediates:
            activations = [x, a1, a2]
            z_values = [z1, z2, z_output]
            return output, activations, z_values
        else:
            return output

    def Training(self, epochs=30, learning_rate=0.001, batch_size=128):

        X_train_flat, y_train, X_test_flat, y_test = LD.getData()

        X_train_flat = X_train_flat.astype(np.float32)
        X_test_flat = X_test_flat.astype(np.float32)

        samples = X_train_flat.shape[0]
        print(f"Training with {samples} examples, {epochs} epoch")

        for epoch in range(epochs):
            indices = np.random.permutation(samples)
            epoch_loss = 0

            for i in range(0, samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train_flat[batch_indices]
                y_batch = y_train[batch_indices]
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate)
                epoch_loss += batch_loss

            if epoch % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / samples:.6f}")

        accuracy = self.evaluate(X_test_flat, y_test)
        print(f"Test-accuracy: {accuracy:.2%}")

    def train_batch(self, X_batch, y_batch, learning_rate):
        batch_size = X_batch.shape[0]

        output, activations, z_values = self.ForwardPass(X_batch, return_intermediates=True)

        # One-hot encoding
        y_one_hot = np.zeros((batch_size, 10), dtype=np.float32)
        for i, label in enumerate(y_batch):
            y_one_hot[i, label] = 1

        error = output - y_one_hot
        batch_loss = np.mean(error ** 2)

        # Backpropagation
        delta_output = error * self.sigmoid_derivative(z_values[2])
        d_output_weights = np.dot(activations[2].T, delta_output) / batch_size
        d_output_bias = np.sum(delta_output, axis=0) / batch_size

        delta_hidden2 = np.dot(delta_output, self.outputwights.T) * self.sigmoid_derivative(z_values[1])
        d_hidden2_weights = np.dot(activations[1].T, delta_hidden2) / batch_size
        d_hidden2_bias = np.sum(delta_hidden2, axis=0) / batch_size

        delta_hidden1 = np.dot(delta_hidden2, self.wights[0].T) * self.sigmoid_derivative(z_values[0])
        d_hidden1_weights = np.dot(activations[0].T, delta_hidden1) / batch_size
        d_hidden1_bias = np.sum(delta_hidden1, axis=0) / batch_size

        self.outputwights -= learning_rate * d_output_weights
        self.outputBias -= learning_rate * d_output_bias

        self.wights[0] -= learning_rate * d_hidden2_weights
        self.hiddenlayersBias[1] -= learning_rate * d_hidden2_bias

        self.inputwights -= learning_rate * d_hidden1_weights
        self.hiddenlayersBias[0] -= learning_rate * d_hidden1_bias

        return batch_loss

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)


    def sigmoid(self, x):
        mask = x >= 0
        result = np.zeros_like(x, dtype=np.float32)

        result[mask] = 1 / (1 + np.exp(-x[mask]))

        exp_x = np.exp(x[~mask])
        result[~mask] = exp_x / (1 + exp_x)

        return result

    def evaluate(self, X_test, y_test):
        output = self.forwardPass(X_test)

        predictions = np.argmax(output, axis=1)
        accuracy = np.sum(predictions == y_test) / len(y_test)

        if accuracy > 0.95:
            print("High accuracy achieved, saving model...")
            np.savez_compressed("model_weights.npz",
                                inputwights=self.inputwights,
                                hiddenlayersBias=self.hiddenlayersBias,
                                wights=self.wights,
                                outputwights=self.outputwights,
                                outputBias=self.outputBias)

        return accuracy
