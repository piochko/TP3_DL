# TP3 : Réseaux de Neurones Convolutifs et Vision par Ordinateur

[cite_start]Ce dépôt contient les travaux réalisés dans le cadre du **Travail Pratique 3** portant sur l'apprentissage profond (Deep Learning) appliqué à la vision par ordinateur[cite: 1, 5]. [cite_start]L'objectif est de maîtriser les architectures CNN, les blocs résiduels et le transfert de style neuronal[cite: 6, 11].

## 📁 Structure du Projet

[cite_start]Le projet est organisé autour des fichiers suivants, conformément aux instructions du TP[cite: 25, 140, 169]:

* [cite_start]**`classification.py`** : Implémentation d'un CNN classique (convolutions, pooling, couches denses) pour la classification d'images sur le dataset CIFAR-10[cite: 57, 115].
* [cite_start]**`resnet.py`** : Implémentation d'une architecture utilisant des **blocs résiduels (ResNets)** avec connexions sauteuses pour faciliter la propagation du gradient[cite: 21, 116].
* [cite_start]**`style.py`** : Script de **transfert de style neuronal** utilisant le modèle VGG16 pré-entraîné pour fusionner le contenu d'une image avec le style d'une autre[cite: 162, 199].
* [cite_start]**`photo.jpg`** : L'image source servant de base pour le contenu[cite: 168].
* [cite_start]**`amelioration.jpg`** : L'image source fournissant le style artistique[cite: 168].
* [cite_start]**`TP3_DL.pdf`** : Rapport final contenant les réponses aux questions théoriques et l'analyse des résultats[cite: 204].

## 🛠️ Dépendances et Installation

[cite_start]Pour exécuter les scripts de ce TP, vous devez installer les bibliothèques suivantes à l'aide de `pip`[cite: 12, 167]:

```bash
pip install tensorflow numpy matplotlib pillow
```
### Détails des bibliothèques :
* **TensorFlow / Keras** : [cite_start]Framework principal pour la construction des modèles CNN et l'utilisation de modèles pré-entraînés comme VGG16[cite: 12, 173].
* **NumPy** : [cite_start]Utilisé pour le chargement et la manipulation des matrices de données (images et labels)[cite: 31].
* **Matplotlib** : [cite_start]Indispensable pour l'affichage des résultats et le traitement visuel[cite: 167].
* **Pillow (PIL)** : [cite_start]Utilisé pour le chargement et le prétraitement des fichiers images externes[cite: 167].

## 🚀 Utilisation

### 1. Classification CIFAR-10
[cite_start]Pour entraîner le modèle CNN classique ou le modèle ResNet sur les 10 classes d'images $32\times32$ du dataset CIFAR-10[cite: 22, 23]:
```bash
python classification.py
# ou
python resnet.py
```
## 2. Transfert de Style

Pour générer une image combinant le contenu de `photo.jpg` et le style de `amelioration.jpg` en utilisant l'extracteur VGG16 :

```bash
python style.py
```
Concepts Abordés

Convolutions et Pooling :
Compréhension du rôle des filtres, du stride et de la réduction de dimensionnalité.

ResNets :
Utilisation de connexions résiduelles (skip connections) pour aider le gradient à se propager dans les réseaux profonds.

Segmentation d'image :
Étude conceptuelle de l'architecture U-Net et des étapes d'upsampling.

Détection d'objets :
Compréhension des Bounding Boxes et de la prédiction des coordonnées (x, y, w, h).
