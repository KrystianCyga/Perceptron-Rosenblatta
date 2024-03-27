import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate, epochs, progMSE, momentum_rate, batch_size):
        self.weights = np.zeros(input_size + 1)  # +1 dla biasu
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.progMSE = progMSE
        self.momentum_rate = momentum_rate
        self.batch_size = batch_size
        self.loss_history = []
        self.weights_history = []  # Lista przechowująca historię zmian wag
        self.prev_delta = np.zeros(input_size + 1)
        self.sum_squared_gradients = None
        self.epsilon = 1e-8

    def activation(self, x):
        return 1 if x >= 0 else 0  # funkcja aktywacji - tutaj używamy prostej funkcji skoku

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # sumowanie wag
        return self.activation(summation)

    def train(self, training_inputs, labels):
        num_samples = len(training_inputs)
        num_batches = num_samples // self.batch_size

        for epoch in range(self.epochs):
            epoch_loss = 0

            for i in range(num_batches):
                batch_inputs = training_inputs[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size: (i + 1) * self.batch_size]

                batch_loss = 0

                for inputs, label in zip(batch_inputs, batch_labels):
                    prediction = self.predict(inputs)
                    error = label - prediction
                    delta = self.learning_rate * error * inputs
                    self.weights[1:] += delta + self.momentum_rate * self.prev_delta[1:]  # Aktualizacja wag z uwzględnieniem momentum
                    self.weights[0] += self.learning_rate * error  # aktualizacja biasu
                    self.prev_delta = delta  # Aktualizacja poprzedniej zmiany wag
                    batch_loss += (error) ** 2

                epoch_loss += batch_loss / self.batch_size

            self.loss_history.append(epoch_loss / num_batches)
            self.weights_history.append(self.weights.copy())  # Zapisanie aktualnych wag
            if epoch_loss < self.progMSE:
                break
            
            # Obliczanie błędu klasyfikacji
            classification_errors = [1 if abs(label - self.predict(inputs)) >= 0.5 else 0 for inputs, label in zip(training_inputs, labels)]
        classification_error = sum(classification_errors) / len(training_inputs)

        return self.loss_history, self.weights_history, classification_errors

# Przykładowe dane treningowe
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])  # funkcja XOR

# Inicjalizacja i uczenie perceptronu
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10, progMSE=0.1, momentum_rate=0.5, batch_size=2)
mse_loss, weights_history, classification_errors = perceptron.train(training_inputs, labels)

# Wykresy
plt.figure(figsize=(18, 6))

# Wykres błędu MSE
plt.subplot(1, 3, 1)
plt.plot(range(1, len(mse_loss)+1), mse_loss, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Mean Squared Error')

# Wykres błędu klasyfikacji
plt.subplot(1, 3, 2)
plt.plot(range(1, len(classification_errors) + 1), classification_errors, marker='o', color='orange')
plt.xlabel('Samples')
plt.ylabel('Classification Error')
plt.title('Classification Error')

# Wykresy wag w obu warstwach
plt.subplot(1, 3, 3)
weights_history = np.array(weights_history)
for i in range(weights_history.shape[1]):  # Dla każdej wagi
    plt.plot(range(1,len(weights_history)+1), weights_history[:, i], marker='o', label=f'Weight {i}')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.title('Weights')
plt.legend()

plt.legend()

plt.tight_layout()
plt.show()
