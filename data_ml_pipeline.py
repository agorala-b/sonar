import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import semantic_dict

warnings.filterwarnings('ignore')

def run_data_ml_pipeline(data_path):
    print("[1/4] Processing Temporal Ontology Features & Running ML Models...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Rev', 'Total_Assets', 'Operating_Cash_Flow'])
    df = df[(df['Rev'] != 0) & (df['Total_Assets'] != 0)]
    df = df.sort_values(by=['Symbol', 'Year']).reset_index(drop=True)

    df['Operating_Income'] = df['Gross_Profit'] - df['Selling_Exp'] - df['Admin_Exp']
    df['Gross_Margin'] = np.where(df['Rev'] != 0, df['Gross_Profit'] / df['Rev'], 0)
    df['Leverage'] = np.where(df['Owners_Equity'] != 0, df['Total_Liabilities'] / df['Owners_Equity'], 0)
    df['Cash_ST_Inv'] = np.maximum(df['ST_Assets'] - df['Rec'] - df['Inv'], 0)
    df['LT_Investments'] = np.maximum(df['Total_Assets'] - df['ST_Assets'] - df['Fixed_Assets'], 0)
    df['Cash'] = df['Cash_ST_Inv']

    for col in ['Rev', 'Rec', 'Inv', 'COGS', 'Gross_Margin', 'Leverage']:
        df[f'{col}_prev'] = df.groupby('Symbol')[col].shift(1)
    df['has_prev_year'] = df['Rev_prev'].notnull()

    epsilon = 1e-9
    df['Delta_Rev'] = np.where((df['Rev_prev'] != 0) & df['has_prev_year'], (df['Rev'] - df['Rev_prev']) / (df['Rev_prev'] + epsilon), 0)
    df['Delta_Rec'] = np.where((df['Rec_prev'] != 0) & df['has_prev_year'], (df['Rec'] - df['Rec_prev']) / (df['Rec_prev'] + epsilon), 0)
    df['Delta_Inv'] = np.where((df['Inv_prev'] != 0) & df['has_prev_year'], (df['Inv'] - df['Inv_prev']) / (df['Inv_prev'] + epsilon), 0)
    df['Delta_COGS'] = np.where((df['COGS_prev'] != 0) & df['has_prev_year'], (df['COGS'] - df['COGS_prev']) / (df['COGS_prev'] + epsilon), 0)
    df['Delta_Gross_Margin'] = df['Gross_Margin'] - df['Gross_Margin_prev']
    df['Delta_Leverage'] = df['Leverage'] - df['Leverage_prev']

    df = df[df['has_prev_year'] == True].copy()

    df['Ratio_Accrual'] = (df['Net_Profit'] - df['Operating_Cash_Flow']) / df['Total_Assets'].replace(0, epsilon)
    df['Flag_Accrual'] = ((df['Net_Profit'] > 0) & (df['Operating_Cash_Flow'] < 0)).astype(int)
    df['Ratio_Channel_Stuffing'] = df['Delta_Rec'] - df['Delta_Rev']
    df['Flag_Channel_Stuffing'] = (df['Ratio_Channel_Stuffing'] > 0.2).astype(int)
    df['Ratio_Noncore'] = (df['Net_Profit'] - df['Operating_Income']) / (np.abs(df['Net_Profit']) + epsilon)
    df['Flag_Noncore'] = (df['Ratio_Noncore'] > 0.5).astype(int)
    df['Flag_Desperation'] = ((df['Delta_Gross_Margin'] < 0) & (df['Delta_Leverage'] > 0.1) & (df['Delta_Inv'] > 0.15)).astype(int)
    df['Ratio_Inventory_Anomaly'] = df['Delta_Inv'] - df['Delta_COGS']
    df['Flag_Inventory_Anomaly'] = (df['Ratio_Inventory_Anomaly'] > 0.2).astype(int)
    df['Ratio_Shadow_Invest'] = df['LT_Investments'] / df['Total_Assets'].replace(0, epsilon)
    df['Flag_Shadow_Invest'] = ((df['Ratio_Shadow_Invest'] > 0.4) & (df['Operating_Cash_Flow'] < 0)).astype(int)
    df['Ratio_Cash_Crunch'] = df['Cash'] / df['ST_Debt'].replace(0, epsilon)
    df['Flag_Cash_Crunch'] = ((df['Short_Term_Ratio'] < 1.0) & (df['Ratio_Cash_Crunch'] < 0.2)).astype(int)
    df['Ratio_Leverage'] = df['Total_Liabilities'] / df['Owners_Equity'].replace(0, epsilon)
    df['Flag_Insolvency'] = ((df['Ratio_Leverage'] > 3.0) | (df['Owners_Equity'] < 0)).astype(int)
    df['Ratio_ROE'] = df['Net_Profit'] / df['Owners_Equity'].replace(0, epsilon)
    df['Flag_Core_Efficiency'] = ((df['Ratio_ROE'] > 0.15) & (df['Operating_Cash_Flow'] > df['Net_Profit'])).astype(int)
    df['Flag_Fortress'] = (df['Cash'] > (df['ST_Debt'] + df['LT_Debt'].fillna(0))).astype(int)

    unique_symbols = df['Symbol'].unique()
    train_symbols, test_symbols = train_test_split(unique_symbols, test_size=0.2, random_state=42)

    train_df = df[df['Symbol'].isin(train_symbols)]
    test_df = df[df['Symbol'].isin(test_symbols)]

    academic_ratios = [
        'Ratio_Accrual', 'Ratio_Channel_Stuffing', 'Ratio_Noncore',
        'Delta_Gross_Margin', 'Delta_Leverage', 'Delta_Inv',
        'Ratio_Inventory_Anomaly', 'Ratio_Shadow_Invest',
        'Ratio_Cash_Crunch', 'Ratio_Leverage', 'Ratio_ROE'
    ]
    flag_cols = [c for c in df.columns if 'Flag_' in c]
    drop_base_cols = ['Target_Fraud', 'Symbol', 'Year', 'has_prev_year']

    X_train_vanilla = train_df.drop(columns=drop_base_cols + flag_cols + academic_ratios, errors='ignore')
    X_test_vanilla = test_df.drop(columns=drop_base_cols + flag_cols + academic_ratios, errors='ignore')
    X_train_hybrid = train_df.drop(columns=drop_base_cols + flag_cols, errors='ignore')
    X_test_hybrid = test_df.drop(columns=drop_base_cols + flag_cols, errors='ignore')

    y_train = train_df['Target_Fraud']
    y_test = test_df['Target_Fraud']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_vanilla_scaled = scaler.fit_transform(imputer.fit_transform(X_train_vanilla))
    X_test_vanilla_scaled = scaler.transform(imputer.transform(X_test_vanilla))
    X_train_hybrid_scaled = scaler.fit_transform(imputer.fit_transform(X_train_hybrid))
    X_test_hybrid_scaled = scaler.transform(imputer.transform(X_test_hybrid))

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
    results = []

    vanilla_models = {
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(scale_pos_weight=ratio, random_state=42, verbose=0)
    }

    for name, model in vanilla_models.items():
        model.fit(X_train_vanilla_scaled, y_train)
        y_pred = model.predict(X_test_vanilla_scaled)
        y_prob = model.predict_proba(X_test_vanilla_scaled)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred) * 100,
            "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
            "F1-Score": f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
            "ROC-AUC": roc_auc_score(y_test, y_prob) * 100
        })

    hybrid_model = XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
    hybrid_model.fit(X_train_hybrid_scaled, y_train)
    y_pred_h = hybrid_model.predict(X_test_hybrid_scaled)
    y_prob_h = hybrid_model.predict_proba(X_test_hybrid_scaled)[:, 1]
    
    results.append({
        "Model": "XGBoost Hybrid",
        "Accuracy": accuracy_score(y_test, y_pred_h) * 100,
        "Recall": recall_score(y_test, y_pred_h, pos_label=1, zero_division=0) * 100,
        "F1-Score": f1_score(y_test, y_pred_h, pos_label=1, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y_test, y_prob_h) * 100
    })

    # M4.5 Apply to all
    X_all_hybrid = df[X_train_hybrid.columns].copy()
    X_all_scaled = scaler.transform(imputer.transform(X_all_hybrid))
    df['xgb_probability'] = hybrid_model.predict_proba(X_all_scaled)[:, 1]
    df['xgb_decision']    = hybrid_model.predict(X_all_scaled)

    # Semantic Output for Test Set (from M2)
    semantic_df = test_df[['Symbol', 'Year', 'Target_Fraud'] + flag_cols].copy()
    semantic_df['AI_Decision'] = y_pred_h
    
    def generate_report_semantic(row):
        red_flags, green_flags = [], []
        green_keys = ['Flag_Core_Efficiency', 'Flag_Fortress']
        for col, meaning in semantic_dict.items():
            if row[col] == 1:
                if col in green_keys: green_flags.append(meaning)
                else: red_flags.append(meaning)
        if row['AI_Decision'] == 1:
            if len(red_flags) > 0: return "CRITICAL WARNING: \n" + "\n".join(red_flags)
            else: return "WARNING: AI detected anomalous financial structure."
        else:
            if len(green_flags) > 0: return "HIGH SAFETY: \n" + "\n".join(green_flags)
            elif len(red_flags) > 0: return "CONCERNING: Safe by AI, but violated: " + red_flags[0]
            else: return "SAFE (Baseline)"
            
    semantic_df['Semantic_Report'] = semantic_df.apply(generate_report_semantic, axis=1)
    sem_map = semantic_df.set_index(['Symbol', 'Year'])['Semantic_Report']
    df['semantic_report'] = df.set_index(['Symbol', 'Year']).index.map(lambda idx: sem_map.get(idx, ''))

    return df, results
