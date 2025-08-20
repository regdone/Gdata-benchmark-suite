import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

print("=== Training CIFAR-10 on GPU ===")

# Explicitly chọn GPU (nếu có)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found!")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Simple CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Build model
model = build_model()

# Training (2 lần để thấy GPU thực sự nhanh hơn)
for run in range(2):
    print(f"\n--- Run {run+1} ---")
    start = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=2)
    end = time.time()
    print(f"Training finished in {end-start:.2f} seconds (GPU, run {run+1})")
