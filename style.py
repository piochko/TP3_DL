import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 1. Configuration et Prétraitement
def load_and_preprocess_image(path, size=(512, 512)):
    try:
        img = Image.open(path).resize(size)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {path} est introuvable. Assurez-vous qu'il est dans le dossier.")
        return None
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    # Inverse le prétraitement VGG16 (centrage sur la moyenne ImageNet)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1] # BGR -> RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 2. Modèle VGG16
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def create_extractor(model, style_layers, content_layers):
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

extractor = create_extractor(vgg, style_layers, content_layers)

# 3. Fonctions de Perte (Loss)
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

@tf.function
def train_step(generated_image, style_targets, content_targets, optimizer, style_weight, content_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(generated_image)
        style_outputs = outputs[:len(style_layers)]
        content_outputs = outputs[len(style_layers):]
        
        # Calcul de la perte de style
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(style_outputs[i]) - gram_matrix(style_targets[i]))**2) 
                               for i in range(len(style_layers))])
        style_loss *= style_weight / len(style_layers)

        # Calcul de la perte de contenu
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[i] - content_targets[i])**2) 
                                 for i in range(len(content_layers))])
        content_loss *= content_weight / len(content_layers)

        total_loss = style_loss + content_loss

    grads = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, -103.939, 151.061)) # Range VGG
    return total_loss

if __name__ == "__main__":
    # Chargement des images spécifiques
    content_img = load_and_preprocess_image('photo.jpg')
    style_img = load_and_preprocess_image('amelioration.jpg')

    if content_img is not None and style_img is not None:
        # Cibles
        style_targets = extractor(style_img)[:len(style_layers)]
        content_targets = extractor(content_img)[len(style_layers):]

        # Image générée (initialisée avec l'image de contenu)
        generated_image = tf.Variable(content_img)

        optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)
        
        # Poids (à ajuster selon vos goûts)
        style_weight = 1e-2  
        content_weight = 1e4 

        print("Début de l'optimisation...")
        for i in range(1, 101):
            loss = train_step(generated_image, style_targets, content_targets, optimizer, style_weight, content_weight)
            if i % 20 == 0:
                print(f"Étape {i}, Perte totale: {loss.numpy():.2f}")

        # Finalisation et sauvegarde
        final_img = deprocess_image(generated_image.numpy())
        plt.figure(figsize=(10, 10))
        plt.imshow(final_img)
        plt.title('Image Générée avec Style Transfer')
        plt.axis('off')
        plt.show()
        
        Image.fromarray(final_img).save('resultat_ameliore.jpg')
        print("Résultat sauvegardé sous 'resultat_ameliore.jpg'")