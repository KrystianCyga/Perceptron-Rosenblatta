import numpy as np
import matplotlib.pyplot as plt

# Funkcja aktywacji sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoidalnej
def sigmoid_derivative(x):
    return x * (1 - x)

# Propagacja w przód
def forward_propagation(x, w1, b1, w2, b2):
    # Pierwsza warstwa
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    # Druga warstwa
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return a1, a2, z1, z2

def train(X, y, epochs, initial_learning_rate, momentum, batch_size):
    w1 = np.array([[2, -2], [-2, 2]], dtype=np.float64)
    b1 = np.array([-1, 3], dtype=np.float64)
    w2 = np.array([[2], [2]], dtype=np.float64)
    b2 = np.array([-3], dtype=np.float64)

    mse_errors = []
    classification_errors = []
    w1_values = []
    w2_values = []
    b1_values = []
    b2_values = []

    w1_velocity = np.zeros_like(w1)
    b1_velocity = np.zeros_like(b1)
    w2_velocity = np.zeros_like(w2)
    b2_velocity = np.zeros_like(b2)

    for epoch in range(epochs):
        # Przemieszanie danych
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Propagacja w przód
            a1, a2, z1, z2 = forward_propagation(X_batch, w1, b1, w2, b2)  # Zwracamy a1, a2, z1, z2
        
            # Obliczanie błędu MSE
            mse_error = np.mean(np.square(y_batch - a2))
            mse_errors.append(mse_error)
            
            # Obliczanie błędu klasyfikacji
            classification_errors_epoch = [1 if abs(label - prediction) >= 0.5 else 0 for label, prediction in zip(y_batch, a2)]
            classification_error = np.mean(classification_errors_epoch)
            classification_errors.append(classification_error)
            
            # Obliczanie gradientu dla błędu
            delta_output = (a2 - y_batch) * sigmoid_derivative(a2)
            delta_hidden = np.dot(delta_output, w2.T) * sigmoid_derivative(a1)
            
            # Aktualizacja współczynnika uczenia
            learning_rate = initial_learning_rate / (1 + epoch * 0.01)
            
            # Aktualizacja wag z momentum
            w2_velocity = momentum * w2_velocity + np.dot(a1.T, delta_output) * learning_rate / X_batch.shape[0]
            w2 -= w2_velocity
            b2_velocity = momentum * b2_velocity + np.sum(delta_output, axis=0) * learning_rate / X_batch.shape[0]
            b2 -= b2_velocity

            w1_velocity = momentum * w1_velocity + np.dot(X_batch.T, delta_hidden) * learning_rate / X_batch.shape[0]
            w1 -= w1_velocity
            b1_velocity = momentum * b1_velocity + np.sum(delta_hidden, axis=0) * learning_rate / X_batch.shape[0]
            b1 -= b1_velocity
            
        w1_values.append(w1.copy())
        w2_values.append(w2.copy())
        b1_values.append(b1.copy())
        b2_values.append(b2.copy())

        if mse_error < 0.005:
            break

    return mse_errors, classification_errors, w1_values, w2_values, b1_values, b2_values


# Dane treningowe
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1],
              [0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0],
              [0], [1], [1], [0],
              [0], [1], [1], [0]])

# Uczenie modelu
epochs = 10000
initial_learning_rate = 0.7
momentum = 0.9
batch_size = 10

mse_errors, classification_errors, w1_values, w2_values, b1_values, b2_values = train(X, y, epochs, initial_learning_rate, momentum,batch_size = 10)

b1_values = np.array(b1_values)
b2_values = np.array(b2_values)

# Wykresy w jednym oknie
plt.figure(figsize=(12, 8))
plt.suptitle(f"Training Results (Initial Learning Rate: {initial_learning_rate}, Epochs: {len(mse_errors)}), Momentum: {momentum}\nWielkosc batcha: {batch_size}", fontsize=16)

# Wykres błędu MSE
plt.subplot(2, 2, 1)
plt.plot(mse_errors)
plt.title('MSE Error')
plt.xlabel('Epoch')
plt.ylabel('Error')

# Wykres błędu klasyfikacji
plt.subplot(2, 2, 2)
plt.plot(classification_errors)
plt.title('Classification Error')
plt.xlabel('Epoch')
plt.ylabel('Error')

# Wykresy wag w obu warstwach
w1_values = np.array(w1_values)
w2_values = np.array(w2_values)
plt.subplot(2, 2, 3)
plt.plot(w1_values[:, 0, 0], label='w11')
plt.plot(w1_values[:, 0, 1], label='w12')
plt.plot(w1_values[:, 1, 0], label='w21')
plt.plot(w1_values[:, 1, 1], label='w22')
for i in range(b1_values.shape[1]):
    plt.plot(b1_values[:, i], label=f'Bias {i+1}')
plt.title('Weights in Layer 1')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(w2_values[:, 0, 0], label='w31')
plt.plot(w2_values[:, 1, 0], label='w32')
plt.plot(b1_values[:, 1], label='Bias')
plt.title('Weights in Layer 2')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.legend()

plt.tight_layout()
plt.show()
