# Data Dictionary — Credit Card Fraud Detection Dataset 2026

- Source: [mohankrishnathalla/credit-card-fraud-detection-dataset-2026](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026)
- Dataset version: 44
- Generated: 2026-11-24 05:49 UTC by `credit_card_fraud_detection_dataset_2026/dataset.py`

| column | group | dtype | description | non_null | n_unique | top_values | min | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| transaction_id | Transaction Identifiers | object | Transaction Identifiers | 100000 | 100000 | TXN-00099984, TXN-00099983, TXN-00099982 | nan | nan | nan |
| customer_id | Transaction Identifiers | object | Transaction Identifiers | 100000 | 8000 | CUS-00004975, CUS-00005784, CUS-00006964 | nan | nan | nan |
| card_id | Transaction Identifiers | object | Transaction Identifiers | 100000 | 8000 | CRD-00004975, CRD-00005784, CRD-00006964 | nan | nan | nan |
| merchant_id | Transaction Identifiers | object | Transaction Identifiers | 100000 | 3000 | MER-00000859, MER-00000198, MER-00001979 | nan | nan | nan |
| transaction_datetime | Transaction Identifiers | object | Transaction Identifiers | 100000 | 99762 | 2026-06-21 14:59:40, 2026-03-29 12:46:15, 2026-03-31 11:07:28 | nan | nan | nan |
| transaction_amount | Transaction Details | float64 | Transaction Details | 100000 | 30862 | nan | 0.5 | 15000.0 | 126.872 |
| merchant_category | Transaction Details | object | Transaction Details | 100000 | 15 | Online Retail, Restaurant & Dining, Grocery & Supermarket | nan | nan | nan |
| merchant_country | Transaction Details | object | Transaction Details | 100000 | 20 | USA, UK, Canada | nan | nan | nan |
| transaction_hour | Transaction Details | int64 | Transaction Details | 100000 | 24 | nan | 0.0 | 23.0 | 14.3939 |
| transaction_day_of_week | Transaction Details | object | Transaction Details | 100000 | 7 | Tuesday, Monday, Thursday | nan | nan | nan |
| card_present | Transaction Details | int64 | Transaction Details | 100000 | 2 | nan | 0.0 | 1.0 | 0.3578 |
| customer_age | Customer Profile | int64 | Customer Profile | 100000 | 68 | nan | 18.0 | 85.0 | 42.2237 |
| customer_gender | Customer Profile | object | Customer Profile | 100000 | 3 | Female, Male, Other | nan | nan | nan |
| customer_state | Customer Profile | object | Customer Profile | 100000 | 40 | CT, NE, AL | nan | nan | nan |
| account_age_days | Customer Profile | int64 | Customer Profile | 100000 | 2217 | nan | 1.0 | 3650.0 | 785.4512 |
| credit_limit | Customer Profile | float64 | Customer Profile | 100000 | 2319 | nan | 500.0 | 36670.0 | 10083.182 |
| credit_score | Customer Profile | int64 | Customer Profile | 100000 | 436 | nan | 328.0 | 850.0 | 682.3293 |
| num_cards_held | Customer Profile | int64 | Customer Profile | 100000 | 5 | nan | 1.0 | 5.0 | 2.1988 |
| transactions_last_1h | Velocity Features | int64 | Velocity Features | 100000 | 9 | nan | 0.0 | 8.0 | 0.7839 |
| transactions_last_24h | Velocity Features | int64 | Velocity Features | 100000 | 19 | nan | 0.0 | 18.0 | 2.9518 |
| transactions_last_7d | Velocity Features | int64 | Velocity Features | 100000 | 40 | nan | 0.0 | 42.0 | 8.8404 |
| amount_last_1h | Velocity Features | float64 | Velocity Features | 100000 | 25927 | nan | 0.0 | 11421.39 | 96.887 |
| amount_last_24h | Velocity Features | float64 | Velocity Features | 100000 | 53453 | nan | 0.0 | 15161.52 | 367.7091 |
| days_since_last_transaction | Velocity Features | float64 | Velocity Features | 100000 | 299 | nan | 0.0 | 47.3 | 3.4964 |
| distance_from_home_km | Risk Signals | float64 | Risk Signals | 100000 | 47545 | nan | 0.0 | 14999.8 | 4316.7531 |
| distance_from_last_tx_km | Risk Signals | float64 | Risk Signals | 100000 | 15070 | nan | 0.0 | 11999.6 | 939.4402 |
| international_transaction | Risk Signals | int64 | Risk Signals | 100000 | 2 | nan | 0.0 | 1.0 | 0.5548 |
| high_risk_country | Risk Signals | int64 | Risk Signals | 100000 | 2 | nan | 0.0 | 1.0 | 0.0493 |
| amount_vs_avg_ratio | Risk Signals | float64 | Risk Signals | 100000 | 10933 | nan | 0.001 | 686.322 | 1.9599 |
| used_chip | Risk Signals | int64 | Risk Signals | 100000 | 2 | nan | 0.0 | 1.0 | 0.3148 |
| used_pin | Risk Signals | int64 | Risk Signals | 100000 | 2 | nan | 0.0 | 1.0 | 0.1839 |
| transaction_channel | Device and Channel | object | Device and Channel | 100000 | 4 | Online, In-store, Mobile App | nan | nan | nan |
| device_type | Device and Channel | object | Device and Channel | 100000 | 4 | Mobile, POS Terminal, Desktop | nan | nan | nan |
| browser | Device and Channel | object | Device and Channel | 58291 | 6 | Chrome, Safari, App | nan | nan | nan |
| is_weekend | Device and Channel | int64 | Device and Channel | 100000 | 2 | nan | 0.0 | 1.0 | 0.2841 |
| avg_transaction_amount_30d | Account Behaviour | float64 | Account Behaviour | 100000 | 6699 | nan | 5.38 | 2000.0 | 124.4025 |
| std_transaction_amount_30d | Account Behaviour | float64 | Account Behaviour | 100000 | 5954 | nan | 2.46 | 1334.92 | 68.462 |
| num_declined_last_30d | Account Behaviour | int64 | Account Behaviour | 100000 | 6 | nan | 0.0 | 5.0 | 0.5129 |
| num_chargebacks_ever | Account Behaviour | int64 | Account Behaviour | 100000 | 5 | nan | 0.0 | 4.0 | 0.3669 |
| email_domain | Account Behaviour | object | Account Behaviour | 100000 | 5 | gmail, corporate, yahoo | nan | nan | nan |
| is_fraud | Targets (3) | int64 | Type: Binary (0 = Legitimate, 1 = Fraud); Fraud rate: 2.06% (2,055 fraud out of 100,000 transactions); Challenge: Class imbalance. Use SMOTE, class weights, or precision-recall curves instead of accuracy.; Expected AUC: 0.90 to 0.95 | 100000 | 2 | nan | 0.0 | 1.0 | 0.0206 |
| fraud_type | Targets (3) | object | Type: 5 classes: Legitimate, Card-Not-Present, Card-Present, Account-Takeover, Synthetic-Identity; Distribution: Legitimate 97.9%, Card-Not-Present 1.1%, Account-Takeover 0.4%, Synthetic-Identity 0.3%, Card-Present 0.2%; Challenge: Severe class imbalance across fraud subtypes. Different feature importance per fraud type.; Expected Macro F1: 0.65 to 0.75 | 100000 | 5 | Legitimate, Card-Not-Present, Account-Takeover | nan | nan | nan |
| fraud_amount_lost | Targets (3) | float64 | Type: Continuous (0.0 for legitimate transactions, USD lost for fraud); Fraud-only mean: $136.18 \| Max: $4,482.16; Challenge: Zero-inflated distribution. Two-stage modelling recommended: first predict fraud, then predict amount.; Expected RMSE (fraud-only): $45 to $65;  | 100000 | 1914 | nan | 0.0 | 4482.16 | 2.7984 |
