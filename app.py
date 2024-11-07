import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Başlık ve Açıklama ---
st.title("Churn Tahmini ve Kampanya Önerisi Uygulaması💸")
st.write("Bu uygulama, müşterilerin churn olup olmayacağını tahmin eder ve kampanya önerileri sunar.")

# --- Veri Yükleme ---
uploaded_file = st.file_uploader("Veri Dosyasını Yükleyin (CSV)", type=["csv"])

# --- Veriyi Hazırlama Fonksiyonu ---
@st.cache_data
def prepare_data(data):
    """Veriyi işler ve eğitim için hazırlar."""
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding
    y = data['Exited']
    return X, y

# --- Model Eğitme Fonksiyonu ---
@st.cache_resource
def train_model(X_train, y_train):
    """Random Forest modelini eğitir ve döndürür."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Kampanya Önerisi Fonksiyonu ---
def suggest_campaign(churn_customers):
    """Kampanya önerilerini üretir."""
    suggestions = churn_customers.copy()
    suggestions['Kampanya Önerisi'] = "Özel Banka Faiz İndirimi"
    return suggestions

# --- Uygulama Akışı ---
if uploaded_file:
    # Veriyi yükleyip ekrana yazdırma
    data = pd.read_csv(uploaded_file)
    st.write("Yüklenen Veri:")
    st.dataframe(data.head())

    # Veriyi hazırla
    X, y = prepare_data(data)

    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modeli eğit
    model = train_model(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Performansı değerlendir
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("Model Performans Raporu:")
    st.json(report)

    # Churn olacak müşterileri belirleme ve kampanya önerisi sunma
    X_test['Churn Tahmini'] = y_pred
    churn_customers = X_test[X_test['Churn Tahmini'] == 1]
    st.subheader("Churn Olacak Müşteriler ve Kampanya Önerileri:")
    suggested_campaigns = suggest_campaign(churn_customers)
    st.dataframe(suggested_campaigns)
else:
    st.write("Lütfen bir CSV dosyası yükleyin.")
