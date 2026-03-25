import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD THE DATA
# ============================================

df = pd.read_csv(r"C:\Users\junio\Downloads\cd\diabetes_012_health_indicators_BRFSS2015.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nTarget distribution (Diabetes_012):")
print(df['Diabetes_012'].value_counts().sort_index())

# ============================================
# 2. CREATE BINARY TARGET
# ============================================

# Create binary target: 1 = Diabetes or Prediabetes, 0 = No Diabetes
df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)

print("\n" + "=" * 60)
print("BINARY TARGET CREATED")
print("=" * 60)
print(f"Diabetes_binary distribution:")
print(f"  0 (No Diabetes): {(df['Diabetes_binary']==0).sum():,}")
print(f"  1 (Diabetes/Prediabetes): {(df['Diabetes_binary']==1).sum():,}")

# ============================================
# 3. PREPARE FEATURES AND TARGET
# ============================================

# Features (all columns except target columns)
X = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)
y = df['Diabetes_binary']

print(f"\nFeatures: {X.shape[1]} columns")
print(f"Feature names: {X.columns.tolist()}")

# ============================================
# 4. SPLIT DATA (NO SAMPLING FOR FULL DATASET)
# ============================================

# Split into train and test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Training class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

# ============================================
# 5. SCALE FEATURES (IMPORTANT FOR NEURAL NETWORKS)
# ============================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (mean=0, std=1)")

# ============================================
# 6. NEURAL NETWORK WITH ReLU ACTIVATION
# ============================================

print("\n" + "=" * 60)
print("BUILDING NEURAL NETWORK WITH ReLU ACTIVATION")
print("=" * 60)

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Build the model
model = keras.Sequential([
    # Input layer + first hidden layer with ReLU
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), 
                 kernel_initializer='he_normal', name='hidden_layer_1'),
    layers.BatchNormalization(),  # Helps with training stability
    layers.Dropout(0.3),  # Regularization to prevent overfitting
    
    # Second hidden layer with ReLU
    layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='hidden_layer_2'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Third hidden layer with ReLU
    layers.Dense(32, activation='relu', kernel_initializer='he_normal', name='hidden_layer_3'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Fourth hidden layer with ReLU (smaller)
    layers.Dense(16, activation='relu', kernel_initializer='he_normal', name='hidden_layer_4'),
    layers.Dropout(0.2),
    
    # Output layer with sigmoid (for binary classification)
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), 
             keras.metrics.Recall(name='recall')]
)

# Display model architecture
model.summary()

# ============================================
# 7. TRAIN THE MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING NEURAL NETWORK")
print("=" * 60)

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# ============================================
# 8. EVALUATE ON TEST SET
# ============================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predict on test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  AUC-ROC: {test_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes/Prediabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              No    Yes")
print(f"Actual No    {cm[0,0]:6,d}  {cm[0,1]:6,d}")
print(f"       Yes   {cm[1,0]:6,d}  {cm[1,1]:6,d}")

# ============================================
# 9. PLOT TRAINING HISTORY
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot training & validation accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot training & validation loss
axes[0, 1].plot(history.history['loss'], label='Training Loss')
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot AUC
axes[1, 0].plot(history.history['auc'], label='Training AUC')
axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
axes[1, 0].set_title('Model AUC')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1, 1].set_title('ROC Curve')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_network_training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Training history plot saved as 'neural_network_training_results.png'")

# ============================================
# 10. COMPARE WITH TREE-BASED MODELS
# ============================================

print("\n" + "=" * 60)
print("COMPARISON WITH TREE-BASED MODELS")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test_scaled)[:, 1])

print(f"\nModel Comparison:")
print(f"{'Model':<25} {'Accuracy':<12} {'AUC-ROC':<12}")
print("-" * 50)
print(f"{'Neural Network (ReLU)':<25} {test_accuracy:<12.4f} {test_auc:<12.4f}")
print(f"{'Random Forest':<25} {rf_accuracy:<12.4f} {rf_auc:<12.4f}")
print(f"{'Gradient Boosting':<25} {gb_accuracy:<12.4f} {gb_auc:<12.4f}")

# ============================================
# 11. VISUALIZE COMPARISON
# ============================================

models = ['Neural Network\n(ReLU)', 'Random Forest', 'Gradient\nBoosting']
accuracies = [test_accuracy, rf_accuracy, gb_accuracy]
aucs = [test_auc, rf_auc, gb_auc]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
bars1 = ax1.bar(models, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# AUC comparison
bars2 = ax2.bar(models, aucs, color=['#2ecc71', '#3498db', '#e74c3c'])
ax2.set_ylabel('AUC-ROC', fontsize=12)
ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Model comparison plot saved as 'model_comparison.png'")
