import pandas as pd
import numpy as np
import time
import joblib
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE # Used for handling imbalance

# --- Configuration ---
MODEL_FILENAME = 'models/fraud_detection_xgb_model.joblib'
SCALER_FILENAME = 'models/scaler.joblib'
RANDOM_STATE = 42
TARGET_COLUMN = 'is_fraud'
NUMERICAL_FEATURES = ['amount', 'hour_of_day', 'distance_from_last_tx', 'count_tx_1h']
PREDICTION_THRESHOLD = 0.80 # The decision threshold for blocking/flagging

# --- Mock Feature Store (Simulating Redis/In-Memory Cache) ---
# Stores the count of transactions in the last hour for each user
FEATURE_STORE = {
    'user_A101': {'count_tx_1h': 2, 'avg_amount': 450.0},
    'user_B202': {'count_tx_1h': 8, 'avg_amount': 12000.0},
    'user_C303': {'count_tx_1h': 1, 'avg_amount': 50.0},
}

# ----------------------------------------------------
# 1. DATA SIMULATION FUNCTION (To replace actual data loading)
# ----------------------------------------------------

def generate_simulated_data(num_samples=20000, fraud_rate=0.005):
    """Generates a synthetic dataset for fraud detection."""
    np.random.seed(RANDOM_STATE)
    
    # Base legitimate transactions
    df = pd.DataFrame({
        'amount': np.random.lognormal(mean=7, sigma=1.5, size=num_samples),
        'hour_of_day': np.random.randint(0, 24, size=num_samples),
        'distance_from_last_tx': np.random.normal(loc=1.5, scale=2.0, size=num_samples).clip(min=0),
        'count_tx_1h': np.random.poisson(lam=2, size=num_samples).clip(max=15),
        TARGET_COLUMN: 0 
    })
    
    # Introduce fraud
    num_fraud = int(num_samples * fraud_rate)
    fraud_indices = np.random.choice(df.index, size=num_fraud, replace=False)
    
    df.loc[fraud_indices, TARGET_COLUMN] = 1
    # Fraudulent patterns: higher amount, high velocity, high distance
    df.loc[fraud_indices, 'amount'] = df.loc[fraud_indices, 'amount'] * np.random.uniform(2, 5) 
    df.loc[fraud_indices, 'count_tx_1h'] = df.loc[fraud_indices, 'count_tx_1h'] + 10 
    df.loc[fraud_indices, 'distance_from_last_tx'] = df.loc[fraud_indices, 'distance_from_last_tx'] + 50
    
    return df

# ----------------------------------------------------
# 2. TRAINING AND EVALUATION PIPELINE
# ----------------------------------------------------

def train_and_evaluate_model(df):
    """Handles data preprocessing, model training, and evaluation."""
    print("--- Starting Training Pipeline ---")
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Separate numerical features for scaling
    X_train_num = X_train[NUMERICAL_FEATURES]
    X_test_num = X_test[NUMERICAL_FEATURES]

    # Initialize and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    # Convert back to DataFrame
    X_train_final = pd.DataFrame(X_train_scaled, columns=NUMERICAL_FEATURES, index=X_train_num.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=NUMERICAL_FEATURES, index=X_test_num.index)

    # Save the scaler for real-time use
    joblib.dump(scaler, SCALER_FILENAME)
    
    # Calculate Imbalance Ratio for Weighting
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight = neg_count / pos_count
    print(f"Fraud Rate: {pos_count / (pos_count + neg_count) * 100:.2f}% | Scale Pos Weight: {scale_pos_weight:.2f}")

    # Initialize and Train XGBoost Classifier
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight, # CRITICAL for imbalance
        use_label_encoder=False, 
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_classifier.fit(X_train_final, y_train)

    # Save the trained model
    joblib.dump(xgb_classifier, MODEL_FILENAME)
    
    # Evaluation
    y_pred_proba = xgb_classifier.predict_proba(X_test_final)[:, 1]
    y_pred = (y_pred_proba >= PREDICTION_THRESHOLD).astype(int)

    print("\n--- Model Evaluation on Test Set ---")
    print(f"Prediction Threshold Used: {PREDICTION_THRESHOLD}")
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Recall (Sensitivity): {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return xgb_classifier, scaler

# ----------------------------------------------------
# 3. REAL-TIME PREDICTION SERVICE MOCK
# ----------------------------------------------------

def load_deployed_assets():
    """Loads the pre-trained model and scaler for the prediction service."""
    try:
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        print("\n--- Deployed Model and Scaler Loaded Successfully ---")
        return model, scaler
    except FileNotFoundError:
        # Create the models directory if it doesn't exist for joblib to save later
        import os
        os.makedirs('models', exist_ok=True)
        print("Error: Model or Scaler file not found. Running training first.")
        return None, None

def get_fraud_score_and_decision(raw_transaction_data, model, scaler):
    """Core function for real-time scoring and decision-making."""
    
    start_time = time.time()
    user_id = raw_transaction_data['user_id']
    
    # 1. Feature Engineering and Lookup (Real-time velocity check)
    velocity_count = FEATURE_STORE.get(user_id, {}).get('count_tx_1h', 1) 
    
    # Assemble feature vector (order must match training data)
    features_dict = {
        'amount': [raw_transaction_data['amount']],
        'hour_of_day': [raw_transaction_data['hour_of_day']],
        'distance_from_last_tx': [raw_transaction_data['distance_from_last_tx']],
        'count_tx_1h': [velocity_count]
    }
    
    X_predict = pd.DataFrame(features_dict)
    
    # 2. Scaling
    X_predict_scaled = scaler.transform(X_predict[NUMERICAL_FEATURES])
    X_predict_final = pd.DataFrame(X_predict_scaled, columns=NUMERICAL_FEATURES)

    # 3. Model Inference
    score = model.predict_proba(X_predict_final)[:, 1][0]
    
    # 4. Decision Engine
    decision = 'ACCEPT'
    if score >= PREDICTION_THRESHOLD:
        decision = 'BLOCK'
    elif score >= (PREDICTION_THRESHOLD * 0.5):
        decision = 'FLAG for Review'
        
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # 5. Mock Feature Store Update (for the next transaction)
    if user_id in FEATURE_STORE:
        FEATURE_STORE[user_id]['count_tx_1h'] += 1
    
    return {
        'score': round(score, 4), 
        'decision': decision, 
        'latency_ms': round(latency_ms, 2)
    }

# ----------------------------------------------------
# 4. MAIN EXECUTION BLOCK
# ----------------------------------------------------

if __name__ == '__main__':
    
    # --- Step 0: Ensure the models directory exists for saving assets ---
    import os
    os.makedirs('models', exist_ok=True)
    
    # --- Step 1: Data Preparation and Training ---
    print("--- 1. Generating Simulated Data ---")
    simulated_df = generate_simulated_data()
    
    # Run training and evaluation
    trained_model, trained_scaler = train_and_evaluate_model(simulated_df)

    # --- Step 2: Real-Time Mock Execution ---
    if trained_model and trained_scaler:
        
        # Load the deployed assets (mocking a service startup)
        model_service, scaler_service = load_deployed_assets()

        print("\n--- 2. Real-Time Transaction Simulation ---")

        # Case A: Normal Transaction (Low risk features)
        tx_normal = {
            'user_id': 'user_A101', 
            'amount': 300.0, 
            'distance_from_last_tx': 0.1, 
            'hour_of_day': 10
        }
        result_normal = get_fraud_score_and_decision(tx_normal, model_service, scaler_service)
        print(f"\n[TX A - Normal] User: {tx_normal['user_id']} | Amount: {tx_normal['amount']}")
        print(f"  Result: Score={result_normal['score']}, Decision='{result_normal['decision']}' (Latency: {result_normal['latency_ms']} ms)")

        # Case B: High-Risk Anomaly (High amount, high distance, high velocity)
        tx_fraud = {
            'user_id': 'user_C303', 
            'amount': 55000.0, 
            'distance_from_last_tx': 150.0, 
            'hour_of_day': 3
        }
        # Manually increase the mock feature store count to simulate high velocity
        FEATURE_STORE['user_C303']['count_tx_1h'] = 15 

        result_fraud = get_fraud_score_and_decision(tx_fraud, model_service, scaler_service)
        print(f"\n[TX B - Fraud] User: {tx_fraud['user_id']} | Amount: {tx_fraud['amount']}")
        print(f"  Result: Score={result_fraud['score']}, Decision='{result_fraud['decision']}' (Latency: {result_fraud['latency_ms']} ms)")