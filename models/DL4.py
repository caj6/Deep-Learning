import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD THE DATA
# ============================================

df = pd.read_csv(r"C:\Users\junio\Downloads\240.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(f"Original shape: {df.shape}")

# ============================================
# 2. REMOVE SSN COLUMN (PRIVACY/COMPLIANCE)
# ============================================

if 'Social_Security_Number' in df.columns:
    df = df.drop('Social_Security_Number', axis=1)
    print(f"\n SSN column removed for GDPR compliance")
    print(f"  New shape: {df.shape}")

# ============================================
# 3. CHECK EXISTING COLUMNS
# ============================================

print("\n" + "=" * 60)
print("COLUMNS IN DATASET")
print("=" * 60)
print(df.columns.tolist())

# ============================================
# 4. VERIFY TARGET VARIABLE
# ============================================

print("\n" + "=" * 60)
print("TARGET VARIABLE: Diabetes_binary")
print("=" * 60)

# Check if Diabetes_binary already exists
if 'Diabetes_binary' in df.columns:
    print("✓ Diabetes_binary column already present")
    print(f"\nDistribution:")
    print(df['Diabetes_binary'].value_counts())
    print(f"\nPercentages:")
    print(df['Diabetes_binary'].value_counts(normalize=True) * 100)
else:
    print("column not found")

# ============================================
# 5. CHECK FOR DUPLICATES
# ============================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# ============================================
# 6. CHECK FOR MISSING VALUES
# ============================================

missing = df.isnull().sum()
print(f"\nMissing values per column:")
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

# ============================================
# 7. DATA TYPES
# ============================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

# ============================================
# 8. DESCRIPTIVE STATISTICS
# ============================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe())

# ============================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================

# Target is 'Diabetes_binary'
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================
# 3. TRAIN/VALIDATION/TEST SPLIT
# ============================================

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\n" + "=" * 60)
print("DATA SPLIT RESULTS")
print("=" * 60)
print(f"Training set:   {X_train.shape[0]:,} samples")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set:       {X_test.shape[0]:,} samples")

# Check class balance in splits
print(f"\nTraining target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nValidation target distribution:\n{y_val.value_counts(normalize=True)}")
print(f"\nTest target distribution:\n{y_test.value_counts(normalize=True)}")    

# ============================================
# 4. NORMALIZE/NUMERICAL FEATURES
# ============================================

# Identify numerical features (non-binary)
numerical_features = ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Income', 'Education']

# Scale numerical features
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"\n✓ Numerical features normalized: {numerical_features}")


# ============================================
# 8. NEURAL NETWORK (Deep Learning)
# ============================================

print("\n" + "=" * 60)
print("NEURAL NETWORK (MLP Classifier)")
print("=" * 60)

# Using sklearn's MLPClassifier as a simpler NN
nn_model = MLPClassifier(
    hidden_layer_sizes=( 1,2,4, 8, 16, 32, 64, 128),  # 7 hidden layers with decreasing neurons
    activation='identity',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)
nn_model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred_nn = nn_model.predict(X_test_scaled)
y_test_proba_nn = nn_model.predict_proba(X_test_scaled)[:, 1]

# ============================================
# 10. NEURAL NETWORK (Deep Learning)
# ============================================

print("\n" + "=" * 60)
print("NEURAL NETWORK (MLP Classifier)")
print("=" * 60)

# Using sklearn's MLPClassifier as a simpler NN
nn_model = MLPClassifier(
    hidden_layer_sizes=( 1,32,64,128),  # 7 hidden layers with decreasing neurons
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
nn_model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred_nn1 = nn_model.predict(X_test_scaled)
y_test_proba_nn1 = nn_model.predict_proba(X_test_scaled)[:, 1]

# ============================================
# 13. NEURAL NETWORK (Deep Learning)
# ============================================

print("\n" + "=" * 60)
print("NEURAL NETWORK (MLP Classifier)")
print("=" * 60)

# Using sklearn's MLPClassifier as a simpler NN
nn_model = MLPClassifier(
    hidden_layer_sizes=( 1,2, 32, 64, 128),  # 7 hidden layers with decreasing neurons
    activation='tanh',
    solver='adam',
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)
nn_model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred_nn2 = nn_model.predict(X_test_scaled)
y_test_proba_nn2 = nn_model.predict_proba(X_test_scaled)[:, 1]

# ============================================
# 8. NEURAL NETWORK (Deep Learning)
# ============================================

print("\n" + "=" * 60)
print("NEURAL NETWORK (MLP Classifier)")
print("=" * 60)

# Using sklearn's MLPClassifier as a simpler NN
nn_model = MLPClassifier(
    hidden_layer_sizes=( 1,4,8,16,32,64, 128),  # 7 hidden layers with decreasing neurons
    activation='logistic',
    solver='sgd',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)
nn_model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred_nn3 = nn_model.predict(X_test_scaled)
y_test_proba_nn3 = nn_model.predict_proba(X_test_scaled)[:, 1]
# ============================================
# 9. MODEL COMPARISON
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

results = pd.DataFrame({
    'Model': ['a', 'b', 'c', 'd'],
    'Test Accuracy': [
        accuracy_score(y_test, y_test_pred_nn),
        accuracy_score(y_test, y_test_pred_nn1),
        accuracy_score(y_test, y_test_pred_nn2),
        accuracy_score(y_test, y_test_pred_nn3)
    ],
    'Test AUC': [
        roc_auc_score(y_test, y_test_proba_nn),
        roc_auc_score(y_test, y_test_proba_nn1),
        roc_auc_score(y_test, y_test_proba_nn2),
        roc_auc_score(y_test, y_test_proba_nn3)
    ],
    'Test Precision': [
        precision_score(y_test, y_test_pred_nn),
        precision_score(y_test, y_test_pred_nn1),
        precision_score(y_test, y_test_pred_nn2),
        precision_score(y_test, y_test_pred_nn3)    
    ],
    'Test Recall': [
        recall_score(y_test, y_test_pred_nn),
        recall_score(y_test, y_test_pred_nn1),
        recall_score(y_test, y_test_pred_nn2),
        recall_score(y_test, y_test_pred_nn3)
    ],
    'Test F1-Score': [
        f1_score(y_test, y_test_pred_nn),
        f1_score(y_test, y_test_pred_nn1),
        f1_score(y_test, y_test_pred_nn2),
        f1_score(y_test, y_test_pred_nn3)
    ]
})

print(results.round(2))

# ============================================
# 10. CONFUSION MATRIX (Best Model)
# ============================================

# Select best model based on AUC
best_model_name = results.loc[results['Test AUC'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")
# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_nn)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\n Confusion matrix saved as 'confusion_matrix.png'")

# ============================================
# 11. ROC CURVE COMPARISON
# ============================================

plt.figure(figsize=(10, 8))

# ROC curves for all models
models_roc = {
    'Neural Network': y_test_proba_nn
}

for name, proba in models_roc.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()

print("\n ROC curves saved as 'roc_curves.png'")
