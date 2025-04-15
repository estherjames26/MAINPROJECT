import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
feature_dir = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\motion_stitcher\\datasorting\\audio_features"
output_model_path = os.path.join(feature_dir, "genre_classifier.pkl")

# Load and combine feature CSVs
splits = ['train', 'val', 'test']
frames = []
for split in splits:
    path = os.path.join(feature_dir, f"{split}_features.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        frames.append(df)

all_data = pd.concat(frames, ignore_index=True)

# Filter for valid entries and 3 or fewer dancers
all_data = all_data[all_data['num_dancers'] <= 3]

# Drop genres with fewer than 10 samples to avoid overfitting

MIN_SAMPLES = 10
genre_counts = all_data['genre'].value_counts()
valid_genres = genre_counts[genre_counts >= MIN_SAMPLES].index
filtered_data = all_data[all_data['genre'].isin(valid_genres)].copy()

# Clean feature columns safely
for col in ['tempo', 'rms', 'onset_strength', 'duration']:
    filtered_data.loc[:, col] = (
        filtered_data[col].astype(str).str.replace('[\[\]]', '', regex=True).astype(float)
    )
# Prepare features and labels
X = filtered_data[['tempo', 'rms', 'onset_strength', 'duration']]
y = filtered_data['genre']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, output_model_path)
print(f" Model saved to: {output_model_path}")

# Evaluation
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues')
plt.title("Genre Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
