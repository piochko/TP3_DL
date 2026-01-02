import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Chargement et Préparation des données CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data() [cite: 23, 33]

NUM_CLASSES = 10 [cite: 35]
INPUT_SHAPE = x_train.shape[1:]  # (32, 32, 3) [cite: 36]

# Normalisation et conversion de type [cite: 38, 39]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Encodage One-Hot des labels [cite: 44, 45, 47]
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# 2. Définition du modèle CNN Amélioré
def build_improved_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=input_shape), [cite: 64, 65]
        layers.BatchNormalization(),
        layers.Activation('relu'), [cite: 64]
        layers.MaxPooling2D(pool_size=(2, 2)), [cite: 68]
        layers.Dropout(0.2),

        # Bloc 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'), [cite: 85, 87]
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)), [cite: 90]
        layers.Dropout(0.3),

        # Bloc 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),

        # Couches Classificatrices
        layers.Flatten(), [cite: 91]
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'), [cite: 93]
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') [cite: 94]
    ])
    return model

model = build_improved_cnn(INPUT_SHAPE, NUM_CLASSES) [cite: 97]

# 3. Compilation et Entraînement
if __name__ == "__main__":
    # Utilisation d'un learning rate légèrement plus faible pour la stabilité
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', [cite: 98]
                  metrics=['accuracy']) [cite: 100]

    print("Entraînement du CNN amélioré...")
    history = model.fit(
        x_train, y_train, [cite: 103]
        batch_size=64, [cite: 104]
        epochs=15, # Augmenté à 15 pour profiter de la BatchNormalization [cite: 105]
        validation_split=0.1, [cite: 106]
        verbose=1
    )

    # 4. Évaluation [cite: 113]
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nPrécision sur le jeu de test : {test_acc:.4f}")