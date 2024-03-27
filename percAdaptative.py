import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate, epochs, progMSE, momentum_rate=0.0, adaptive_learning=True, epsilon=1e-8):
        self.weights = np.zeros(input_size + 1)  # +1 dla biasu
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.progMSE = progMSE
        self.loss_history = []
        self.weights_history = []  # Lista przechowująca historię zmian wag
        self.momentum_rate = momentum_rate
        self.prev_delta = np.zeros(input_size + 1)
        self.adaptive_learning = adaptive_learning
        self.epsilon = epsilon
        self.sum_squared_gradients = None
        self.adapt_history = []

    def activation(self, x):
        return 1 if x >= 0 else 0  # funkcja aktywacji - tutaj używamy prostej funkcji skoku

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # sumowanie wag
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                delta = self.learning_rate * error * inputs
                
                # Aktualizacja wag z uwzględnieniem momentum
                delta += self.momentum_rate * self.prev_delta[1:]

                if self.adaptive_learning:
                    # Aktualizacja współczynnika uczenia zgodnie z algorytmem Adagrad
                    if self.sum_squared_gradients is None:
                        self.sum_squared_gradients = np.zeros_like(self.weights)
                    self.sum_squared_gradients[1:] += delta[1:] ** 2  # Poprawka 
                    adjusted_learning_rate = self.learning_rate / (np.sqrt(self.sum_squared_gradients[1:]) + self.epsilon)
                    delta[1:] = delta[1:] * adjusted_learning_rate[1:]  # Waga 0 (bias) nie jest aktualizowana adaptacyjnie

                self.weights[1:] += delta
                self.weights[0] += self.learning_rate * error  # aktualizacja biasu
                self.prev_delta = delta  # Aktualizacja poprzedniej zmiany wag
                epoch_loss += (error) ** 2
            self.adapt_history.append(adjusted_learning_rate)
            self.loss_history.append(epoch_loss / len(labels))
            self.weights_history.append(self.weights.copy())  # Zapisanie aktualnych wag
            if epoch_loss < self.progMSE:
                break

        # Obliczanie błędu klasyfikacji
        classification_errors = [1 if abs(label - self.predict(inputs)) >= 0.5 else 0 for inputs, label in zip(training_inputs, labels)]
        classification_error = sum(classification_errors) / len(training_inputs)

        return self.loss_history, classification_errors, self.weights_history, self.adapt_history

# Przykładowe dane treningowe
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])  # funkcja XOR

# Inicjalizacja i uczenie perceptronu z adaptacyjnym współczynnikiem uczenia (Adagrad)
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10, progMSE=0.1, adaptive_learning=True)
mse_loss, classification_errors, weights_history,adapt_h = perceptron.train(training_inputs, labels)

# Wykresy
plt.figure(figsize=(18, 6))

# Wykres błędu MSE
plt.subplot(1, 4, 1)
plt.plot(range(1, len(mse_loss)+1), mse_loss, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Mean Squared Error')

# Wykres błędu klasyfikacji
plt.subplot(1, 4, 2)
plt.plot(range(1, len(classification_errors) + 1), classification_errors, marker='o', color='orange')
plt.xlabel('Samples')
plt.ylabel('Classification Error')
plt.title('Classification Error')

# Wykresy wag w obu war

plt.subplot(1, 4, 3)
weights_history = np.array(weights_history)
for i in range(weights_history.shape[1]):  # Dla każdej wagi
    plt.plot(range(1,len(weights_history)+1), weights_history[:, i], marker='o', label=f'Weight {i}')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.title('Weights')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(range(1, len(weights_history) + 1), adapt_h, marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Adaptative learning rate')
plt.title('Adaptative learning rate')
plt.yticks(np.arange(0.5, 1, 0.05))

plt.tight_layout()
plt.show()