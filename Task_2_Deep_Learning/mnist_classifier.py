# MNIST Digit Classifier using a basic Neural Network

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0 to 1 range)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Flatten 28x28 images to 1D
    Dense(128, activation='relu'),        # First hidden layer
    Dense(64, activation='relu'),         # Second hidden layer
    Dense(10, activation='softmax')       # Output layer for 10 digits
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 5 epochs
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save("mnist_trained_model.h5")
print("Saved model to 'mnist_trained_model.h5'")
