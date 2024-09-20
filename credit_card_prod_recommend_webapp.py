import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title('Credit Card Predictor Scoring App')

# Upload the data
uploaded_file = st.file_uploader("Upload the credit card customer dataset (Excel file)", type="xlsx")

if uploaded_file:
    # Load the dataset
    data_df = pd.read_excel(uploaded_file, dtype={'CUSTOMER_NO': str})
    st.write("Dataset preview:", data_df.head())

    # Features and target
    features = ["TOTAL_TXNS", "TOTAL_TXN_AMOUNT", "AVG_TXN_AMOUNT"]
    X = data_df[features]
    y = data_df["HAS_CREDIT_CARD"]

    # Display class imbalance
    class_distribution = y.value_counts(normalize=True)
    st.write(f"Class distribution:\n{class_distribution}")

    # Feature Scaling
    scaler = StandardScaler()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost Classifier
    # xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    # xgb_model.fit(X_train_scaled, y_train)

    # Hyperparameter tuning for XGBoost
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 2, 3]  # Adjust based on class imbalance
    }
    xgb_grid = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"), xgb_params, cv=5, scoring='roc_auc')
    xgb_grid.fit(X_train, y_train)

    # Best XGBoost Model
    best_xgb_model = xgb_grid.best_estimator_



    # Predict the propensity scores for all customers
    data_df['propensity_score'] = best_xgb_model.predict_proba(scaler.transform(X))[:, 1]

    # Evaluate the model on the test set
    y_pred_proba_xgb = best_xgb_model.predict_proba(X_test_scaled)[:, 1]
    roc_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

    st.write(f'XGBoost ROC AUC score: {roc_score_xgb * 100:.2f}')

    # Display confusion matrix and classification report
    y_pred = best_xgb_model.predict(X_test_scaled)
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:", classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_xgb)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_score_xgb)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Generate product recommendations
    threshold = st.slider('Select the threshold for high propensity score', 0.5, 1.0, 0.7)
    potential_customers = data_df[(data_df['HAS_CREDIT_CARD'] == 0) & (data_df['propensity_score'] > threshold)]

    st.write(f"Recommend product to {len(potential_customers)} customers based on propensity scores.")

    # Output the customer IDs who are good candidates for product recommendation
    recommended_customers = potential_customers['CUSTOMER_NO']
    st.write("Recommended Customers' IDs:", recommended_customers)

    # Optionally download recommended customers as CSV
    if st.button('Download Recommended Customers as CSV'):
        recommended_customers.to_csv('recommended_customers.csv', index=False)
        st.write("Download complete!")
