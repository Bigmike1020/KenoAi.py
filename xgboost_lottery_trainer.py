
import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

# Load your dataset
df = pd.read_csv("Results_01052025 (6).csv")

# Preprocess
df = df.sort_values("Draw Number").reset_index(drop=True)
ball_columns = [col for col in df.columns if col.startswith("Ball")]

# Parameters
max_ball_number = 80
window_size = 20

# Initialize
X = []
y = []
last_seen = {n: -1 for n in range(1, max_ball_number + 1)}

# Feature engineering loop
for i in range(window_size, len(df)):
    past_draws = df.iloc[i - window_size:i]
    current_draw = df.iloc[i]

    # Frequency feature
    freq = Counter(past_draws[ball_columns].values.flatten())
    freq_feature = [freq.get(n, 0) for n in range(1, max_ball_number + 1)]

    # Recency feature
    recency_feature = [i - last_seen[n] if last_seen[n] != -1 else window_size + 1 for n in range(1, max_ball_number + 1)]

    # Combine features
    combined_feature = freq_feature + recency_feature
    X.append(combined_feature)

    # Target
    actual_numbers = set(current_draw[ball_columns].values)
    target_vector = [1 if n in actual_numbers else 0 for n in range(1, max_ball_number + 1)]
    y.append(target_vector)

    # Update last seen
    for num in actual_numbers:
        last_seen[num] = i

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model per number
f1_scores = {}
models = {}

for n in tqdm(range(max_ball_number), desc="Training XGBoost models"):
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train[:, n])
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test[:, n], y_pred)
    
    f1_scores[n + 1] = f1  # Ball number starts at 1
    models[n + 1] = model

# Save results
with open("xgb_f1_scores.pkl", "wb") as f:
    pickle.dump(f1_scores, f)

with open("xgb_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("Training complete. F1 scores saved to 'xgb_f1_scores.pkl'. Models saved to 'xgb_models.pkl'.")
