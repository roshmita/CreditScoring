# Credit Risk Scoring Using XGBoost and SHAP

## 📌 Project Overview
This project applies supervised machine learning models to predict the probability of credit default using structured financial data. The main objective is to build an interpretable and effective classifier for credit risk assessment, with a focus on model performance and transparency using SHAP.

## 📊 Dataset Summary
- **Source**: UCI German Credit dataset
- **Observations**: 1000 customers
- **Target**: Creditworthiness (1 = Good, 0 = Bad)
- **Features**:
  - Categorical: Purpose, Job, Housing, etc.
  - Numerical: CreditAmount, Duration, Age, etc.
- **Imbalance**: Approximately 70% good credit, 30% bad credit

## 📈 Key Metrics and Results
- **Models Trained**: Logistic Regression, Random Forest, XGBoost
- **Best ROC AUC**: 
  - Random Forest: ~0.78
  - XGBoost (tuned): ~0.76
- **Class 0 (Defaulters) Performance**:
  - Recall: up to 0.80+
  - F1-score improved after threshold tuning
- **Evaluation Tools**: ROC curve, confusion matrix, classification report

## 🧠 Model Interpretation Visuals
- Used `SHAP` (SHapley Additive exPlanations) for local and global interpretability.
- **SHAP Summary Plots**: Show top contributing features (CreditAmount, Duration, Age, Savings).
- **Dependence Plots**: Visualize non-linear feature effects and interactions.
- **Class-wise SHAP**: Separate SHAP plots for defaulters and non-defaulters enhance transparency.

## 💼 Business Implications
- Enables data-driven risk segmentation for loan applicants.
- Helps reduce financial losses by correctly flagging high-risk customers.
- SHAP explanations provide compliance-ready justifications for credit decisions.
- Can be integrated into a larger loan approval pipeline or risk scoring engine.

---

> Developed using Python, scikit-learn, XGBoost, SHAP, and Matplotlib. All visuals and outputs are aligned with business interpretability needs.
