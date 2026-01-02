import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Définition du Bloc Résiduel Amélioré
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x
    
    # Premier bloc : Conv -> BatchNormalization -> ReLU
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_initializer='he_normal')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    
    # Deuxième bloc : Conv -> BatchNormalization
    y = layers.Conv2D(filters, kernel_size, padding='same', 
                      kernel_initializer='he_normal')(y)
    y = layers.BatchNormalization()(y)
    
    # Ajustement du raccourci (skip connection) si les dimensions changent [cite: 21, 131, 132]
    if stride > 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, 
                                 padding='same', kernel_initializer='he_normal')(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Addition et activation finale [cite: 135, 137]
    z = layers.Add()([shortcut, y])
    z = layers.Activation('relu')(z)
    return z

# 2. Construction du Mini-ResNet
def build_mini_resnet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Couche d'entrée initiale
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Empilement de blocs résiduels [cite: 140, 143, 145, 147]
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2) # Réduction de dimension (16x16)
    x = residual_block(x, 128, stride=2) # Réduction de dimension (8x8)
    
    # Couche de sortie
    x = layers.GlobalAveragePooling2D()(x) # Plus efficace que Flatten pour les ResNets
    x = layers.Dropout(0.3)(x) # Anti-surapprentissage
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# 3. Préparation et Entraînement
if __name__ == "__main__":
    # Paramètres [cite: 35, 36]
    NUM_CLASSES = 10
    INPUT_SHAPE = (32, 32, 3)
    
    # Chargement CIFAR-10 [cite: 32, 33]
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalisation [cite: 38, 42]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Labels en One-Hot [cite: 44, 45, 47]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    model = build_mini_resnet(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Résumé du Mini-ResNet amélioré :")
    model.summary()
    
    # Entraînement [cite: 101, 102, 105, 106]
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=15, # Augmenté légèrement pour voir l'effet du ResNet
        validation_split=0.1
    )
    
    # Évaluation [cite: 113]
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nPrécision finale sur le jeu de test : {test_acc:.4f}")