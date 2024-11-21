Venv\Scripts\activateimport streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def detect_categorical_columns(df):
    return df.select_dtypes(include=[ 'object', 'category' ]).columns.tolist()

# Streamlit başlığı
st.title("Churn Tahmini ve Kampanya Önerisi Uygulaması💸")
st.write("Bu uygulama, müşterilerin churn olup olmayacağını tahmin eder ve kampanya önerileri sunar.")


def load_and_preprocess_data(df):
    # Gereksiz sütunları kaldır
    columns_to_drop = [ 'CustomerId', 'Surname', 'RowNumber' ]  # Exited'ı burada kaldırmıyoruz, hedef değişken
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Tenure Months = 0 olanları "0-1 years" grubuna ekle
    df.loc [ df [ 'Tenure_Months' ] == 0, 'Tenure_Group' ] = '0-1 years'

    # Yeni Tenure Yılı
    df [ 'NEW_Tenure_Years' ] = df [ 'Tenure_Months' ] / 12

    # Ürün başına Tenure
    df [ 'NEW_Product_per_Tenure_Years' ] = df [ 'NEW_Tenure_Years' ] / df [ 'NumOfProducts' ]

    # Credit Score Mapping
    credit_map = {'Good': 1, 'Fair': 2, 'Excellent': 3}
    df [ 'Credit_Score_Mapped' ] = df [ 'Credit_Score_Category' ].map(credit_map)

    # Active Credit Card Encoding
    df [ 'Active_Credit_Card' ] = (df [ 'HasCrCard' ] == 1).astype(int)

    # Eksik değer temizleme
    df.dropna(inplace=True)
    return df


def encode_categorical(df, categorical_columns):
    valid_columns = [ col for col in categorical_columns if col in df.columns ]
    if not valid_columns:
        raise ValueError("No valid categorical columns found in the dataset.")

    encoder = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first'), valid_columns)
        ],
        remainder='passthrough'
    )

    df_encoded = encoder.fit_transform(df)
    feature_names = encoder.get_feature_names_out()
    return pd.DataFrame(df_encoded, columns=feature_names)


def train_with_best_params(X_train, y_train, model_name, smote):
    pipeline = setup_model_pipeline(model_name, smote)
    param_grid = get_hyperparameters(model_name)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',  # F1 skoru ile değerlendirme
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def setup_model_pipeline(model_name, smote):
    models = {
        'random_forest': RandomForestClassifier(),
        'xgboost': XGBClassifier(),
        'lightgbm': LGBMClassifier(),
        'catboost': CatBoostClassifier(verbose=0)
    }
    model = models.get(model_name, RandomForestClassifier())
    if smote:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE()),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    return pipeline


def get_hyperparameters(model_name):
    return {
        'random_forest': {
            'model__n_estimators': [ 100, 200 ],
            'model__max_depth': [ None, 10 ],
            'model__min_samples_split': [ 2, 5 ]
        },
        'xgboost': {
            'model__learning_rate': [ 0.01, 0.1 ],
            'model__max_depth': [ 3, 5 ],
            'model__n_estimators': [ 100, 200 ]
        },
        'lightgbm': {
            'model__learning_rate': [ 0.01, 0.1 ],
            'model__num_leaves': [ 31, 50 ],
            'model__n_estimators': [ 100, 200 ]
        },
        'catboost': {
            'model__learning_rate': [ 0.01, 0.1 ],
            'model__depth': [ 4, 6 ],
            'model__iterations': [ 100, 200 ]
        }
    }.get(model_name, {})


def split_data(X, y, test_size=0.1):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)



uploaded_file = st.file_uploader("Upload your dataset", type=[ "csv" ])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    categorical_features = detect_categorical_columns(df)
    st.write("Automatically detected categorical columns:", categorical_features)
    smote = st.checkbox("Use SMOTE for class balancing", value=True)
    model_name = st.selectbox("Select a model", [ "random_forest", "xgboost", "lightgbm", "catboost" ], index=0)
    if st.button("Run Pipeline"):
        df = load_and_preprocess_data(df)
        y = df [ 'Exited' ]  # Exited hedef değişkeni
        X = df.drop(columns=[ 'Exited' ])  # Exited hariç diğer özellikler
        if categorical_features:
            X = encode_categorical(X, categorical_features)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.1)
        best_pipeline, best_params, best_score = train_with_best_params(X_train, y_train, model_name, smote)
        st.write(f"Best Parameters: {best_params}")
        st.write(f"Best Cross-Validated F1 Score (Train Set): {best_score}")
        predictions = best_pipeline.predict(X_test)
        probabilities = best_pipeline.predict_proba(X_test) [ :, 1 ]

        # Tahmin edilen churn sayısı
        churn_count = sum(predictions)
        st.write(f"Total Predicted Churn Customers: {churn_count}")

        report = classification_report(y_test, predictions, output_dict=True)
        st.write("Classification Report on Test Set:")
        st.json(report)
        roc_auc = roc_auc_score(y_test, probabilities)
        st.write(f"ROC-AUC Score on Test Set: {roc_auc}")
