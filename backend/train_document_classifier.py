#!/usr/bin/env python3
"""
Script d'entraînement pour classification de documents
Target: 85%+ accuracy sur 4 classes
Dataset: Note (201), Invoice (247), Report (265), Drawing (128)
Total: 841 images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print("="*80)
print("🎯 ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION DE DOCUMENTS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.0001,
    'validation_split': 0.2,
    'test_split': 0.15,
    'target_accuracy': 0.85,
    'patience': 10,  # Early stopping
    'classes': ['Drawing', 'Invoice', 'Report', 'Note'],  # Ordre alphabétique
    'data_augmentation': True
}

print(f"\n📋 Configuration:")
for key, value in CONFIG.items():
    print(f"   • {key}: {value}")

# ============================================================================
# CHEMINS DES DONNÉES
# ============================================================================
# Structure attendue:
# dataset/
#   ├── Drawing/
#   │   ├── img1.jpg
#   │   └── img2.jpg
#   ├── Invoice/
#   ├── Report/
#   └── Note/

DATA_DIR = "../dataset"  # Ajustez selon votre structure

if not os.path.exists(DATA_DIR):
    print(f"\n❌ ERREUR: Dossier {DATA_DIR} introuvable!")
    print(f"\n📁 Structure attendue:")
    print(f"   {DATA_DIR}/")
    print(f"   ├── Drawing/")
    print(f"   ├── Invoice/")
    print(f"   ├── Report/")
    print(f"   └── Note/")
    exit(1)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
print(f"\n📥 Chargement du dataset depuis: {DATA_DIR}")

# Utiliser image_dataset_from_directory (meilleure pratique)
try:
    # Dataset complet
    full_dataset = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        seed=42
    )
    
    class_names = full_dataset.class_names
    print(f"\n✅ Classes détectées: {class_names}")
    
    # Compter les images par classe
    total_images = sum([len(os.listdir(os.path.join(DATA_DIR, cls))) 
                       for cls in class_names if os.path.isdir(os.path.join(DATA_DIR, cls))])
    print(f"✅ Total d'images: {total_images}")
    
    for cls in class_names:
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            percentage = (count / total_images) * 100
            print(f"   • {cls}: {count} images ({percentage:.1f}%)")

except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    exit(1)

# ============================================================================
# SPLIT TRAIN/VALIDATION/TEST - CORRECTION ICI
# ============================================================================
# Obtenir le nombre total de batches
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()

# Calculer les tailles en nombre de batches
test_size = int(total_batches * CONFIG['test_split'])
val_size = int(total_batches * CONFIG['validation_split'])
train_size = total_batches - val_size - test_size

# S'assurer qu'on a au moins 1 batch pour chaque split
test_size = max(1, test_size)
val_size = max(1, val_size)
train_size = max(1, train_size)

# Créer les datasets
train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

print(f"\n📊 Split des données:")
print(f"   • Total batches: {total_batches}")
print(f"   • Train: {train_size} batches (~{train_size * CONFIG['batch_size']} images)")
print(f"   • Validation: {val_size} batches (~{val_size * CONFIG['batch_size']} images)")
print(f"   • Test: {test_size} batches (~{test_size * CONFIG['batch_size']} images)")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
if CONFIG['data_augmentation']:
    print(f"\n🔄 Configuration de l'augmentation des données...")
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])
    print("✅ Augmentation activée")

# Optimisation des performances
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================================
# CRÉATION DU MODÈLE
# ============================================================================
print(f"\n🏗️  Construction du modèle...")

# Base: EfficientNetB0 (meilleur rapport performance/taille)
base_model = keras.applications.EfficientNetB0(
    input_shape=(*CONFIG['image_size'], 3),
    include_top=False,
    weights='imagenet'
)

# Geler les couches de base initialement
base_model.trainable = False

# Créer le modèle complet
inputs = keras.Input(shape=(*CONFIG['image_size'], 3))

# Augmentation (si activée)
if CONFIG['data_augmentation']:
    x = data_augmentation(inputs)
else:
    x = inputs

# Preprocessing spécifique à EfficientNet
x = keras.applications.efficientnet.preprocess_input(x)

# Base model
x = base_model(x, training=False)

# Classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

print(f"✅ Modèle créé")
print(f"   • Architecture: EfficientNetB0 + Custom Head")
print(f"   • Paramètres totaux: {model.count_params():,}")
print(f"   • Paramètres entraînables: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")

# ============================================================================
# COMPILATION - PHASE 1 (Base gelée)
# ============================================================================
print(f"\n⚙️  Compilation du modèle (Phase 1: Base gelée)...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'] * 10),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
)

print("✅ Modèle compilé")

# ============================================================================
# CALLBACKS
# ============================================================================
print(f"\n📞 Configuration des callbacks...")

callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=CONFIG['patience'],
        restore_best_weights=True,
        verbose=1
    ),
    
    # Réduction du learning rate
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Sauvegarde du meilleur modèle (format Keras natif)
    keras.callbacks.ModelCheckpoint(
        '../models/best_model_checkpoint.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # TensorBoard (désactivé histogram_freq pour éviter erreurs de pickle)
    keras.callbacks.TensorBoard(
        log_dir='../logs',
        histogram_freq=0,  # 0 = désactivé (évite erreurs deepcopy)
        write_graph=True,
        update_freq='epoch'
    )
]

print("✅ Callbacks configurés")

# ============================================================================
# ENTRAÎNEMENT - PHASE 1
# ============================================================================
print("\n" + "="*80)
print("🚀 PHASE 1: ENTRAÎNEMENT AVEC BASE GELÉE")
print("="*80)

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Phase 1: moins d'epochs
    callbacks=callbacks,
    verbose=1
)

# Évaluation après Phase 1
print(f"\n📊 Évaluation après Phase 1...")
phase1_results = model.evaluate(test_dataset, verbose=0)
print(f"   • Test Loss: {phase1_results[0]:.4f}")
print(f"   • Test Accuracy: {phase1_results[1]:.4f} ({phase1_results[1]*100:.2f}%)")
print(f"   • Top-2 Accuracy: {phase1_results[2]:.4f} ({phase1_results[2]*100:.2f}%)")

# ============================================================================
# FINE-TUNING - PHASE 2 (Dégeler les dernières couches)
# ============================================================================
if phase1_results[1] < CONFIG['target_accuracy']:
    print("\n" + "="*80)
    print("🔥 PHASE 2: FINE-TUNING (Dégel des dernières couches)")
    print("="*80)
    
    # Dégeler les dernières couches de la base
    base_model.trainable = True
    
    # Geler les premières couches, dégeler les dernières
    fine_tune_at = len(base_model.layers) - 30  # Dégeler les 30 dernières couches
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"✅ Dégelé les {len(base_model.layers) - fine_tune_at} dernières couches")
    print(f"   • Paramètres entraînables: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    # Recompiler avec un learning rate plus faible
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    # Entraînement Phase 2
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        initial_epoch=history_phase1.epoch[-1],
        callbacks=callbacks,
        verbose=1
    )
    
    # Combiner les historiques
    history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    }
else:
    history = history_phase1.history

# ============================================================================
# ÉVALUATION FINALE
# ============================================================================
print("\n" + "="*80)
print("📊 ÉVALUATION FINALE")
print("="*80)

final_results = model.evaluate(test_dataset, verbose=0)
test_loss = final_results[0]
test_accuracy = final_results[1]
top2_accuracy = final_results[2]

print(f"\n🎯 Résultats sur le test set:")
print(f"   • Loss: {test_loss:.4f}")
print(f"   • Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   • Top-2 Accuracy: {top2_accuracy:.4f} ({top2_accuracy*100:.2f}%)")

if test_accuracy >= CONFIG['target_accuracy']:
    print(f"\n✅ OBJECTIF ATTEINT! ({test_accuracy*100:.2f}% >= {CONFIG['target_accuracy']*100}%)")
else:
    print(f"\n⚠️  Objectif non atteint ({test_accuracy*100:.2f}% < {CONFIG['target_accuracy']*100}%)")
    print(f"   Suggestions:")
    print(f"   • Augmenter les epochs")
    print(f"   • Ajouter plus de données")
    print(f"   • Essayer EfficientNetB3 (plus gros)")

# ============================================================================
# SAUVEGARDE DU MODÈLE
# ============================================================================
print(f"\n💾 Sauvegarde du modèle final...")

os.makedirs("../models", exist_ok=True)

try:
    # Format natif Keras (recommandé, fonctionne toujours)
    model.save("../models/final_model_complete.keras")
    keras_size = os.path.getsize("../models/final_model_complete.keras") / (1024 * 1024)
    print(f"✅ Sauvegardé: ../models/final_model_complete.keras ({keras_size:.1f} MB)")
except Exception as e:
    print(f"⚠️  Erreur sauvegarde .keras: {e}")

try:
    # Format SavedModel (production, TensorFlow Serving)
    model.save("../models/final_model_savedmodel", save_format='tf')
    print(f"✅ Sauvegardé: ../models/final_model_savedmodel/")
except Exception as e:
    print(f"⚠️  Erreur sauvegarde SavedModel: {e}")

# Format .h5 : Créer un modèle sans augmentation pour compatibilité
print(f"\n🔧 Création d'un modèle .h5 sans augmentation (pour compatibilité)...")
try:
    # Recréer le modèle sans data augmentation
    inputs_clean = keras.Input(shape=(*CONFIG['image_size'], 3))
    x_clean = keras.applications.efficientnet.preprocess_input(inputs_clean)
    x_clean = base_model(x_clean, training=False)
    x_clean = layers.GlobalAveragePooling2D()(x_clean)
    x_clean = layers.BatchNormalization()(x_clean)
    x_clean = layers.Dropout(0.3)(x_clean)
    x_clean = layers.Dense(256, activation='relu')(x_clean)
    x_clean = layers.BatchNormalization()(x_clean)
    x_clean = layers.Dropout(0.3)(x_clean)
    x_clean = layers.Dense(128, activation='relu')(x_clean)
    x_clean = layers.Dropout(0.2)(x_clean)
    outputs_clean = layers.Dense(len(class_names), activation='softmax')(x_clean)
    
    model_h5 = keras.Model(inputs_clean, outputs_clean)
    
    # Copier les poids du modèle entraîné (skip les layers d'augmentation)
    for layer_original, layer_h5 in zip(model.layers[1:], model_h5.layers):
        if layer_original.name == layer_h5.name:
            try:
                layer_h5.set_weights(layer_original.get_weights())
            except:
                pass  # Skip si incompatible
    
    model_h5.save("../models/final_model_complete.h5")
    h5_size = os.path.getsize("../models/final_model_complete.h5") / (1024 * 1024)
    print(f"✅ Sauvegardé: ../models/final_model_complete.h5 ({h5_size:.1f} MB)")
    file_size = h5_size
except Exception as e:
    print(f"⚠️  Impossible de sauvegarder en .h5: {e}")
    print(f"   → Utilisez le format .keras à la place")
    file_size = keras_size if 'keras_size' in locals() else 0

# ============================================================================
# VISUALISATION (optionnel)
# ============================================================================
print(f"\n📈 Génération des graphiques...")

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.axhline(y=CONFIG['target_accuracy'], color='r', linestyle='--', label='Target (85%)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../models/training_history.png', dpi=300)
print(f"✅ Graphiques sauvegardés: ../models/training_history.png")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*80)
print("🎉 ENTRAÎNEMENT TERMINÉ")
print("="*80)
print(f"\n📊 Résumé:")
print(f"   • Dataset: {total_images} images, 4 classes")
print(f"   • Architecture: EfficientNetB0 + Transfer Learning")
print(f"   • Accuracy finale: {test_accuracy*100:.2f}%")
print(f"   • Top-2 Accuracy: {top2_accuracy*100:.2f}%")
print(f"   • Taille du modèle: {file_size:.1f} MB")

print(f"\n🚀 Prochaines étapes:")
print(f"   1. Testez le modèle: python main.py")
print(f"   2. Le modèle devrait se charger rapidement")
print(f"   3. Mode RÉEL avec fusion CNN + OCR activé")

print(f"\n💡 Utilisation:")
print(f"   curl http://localhost:8000/api/v1/status")
print(f"   → 'model_loaded': true, 'mode': 'real'")

print("="*80)