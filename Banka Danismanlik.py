import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Banka Danismanlik Hizmeti/Customer Churn Classification.csv"
data = pd.read_csv(file_path)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

data.head()
data.info()

data.isnull().sum()

# Tenure_Group sütununda en sık tekrarlanan kategoriyi bulup onunla dolduruyoruz
most_frequent_value = data['Tenure_Group'].mode()[0]
print(f"En Sık Görülen Kategori: {most_frequent_value}")

data['Tenure_Group'].fillna(most_frequent_value, inplace=True)
data.isnull().sum()

# Cinsiyet ve müşteri kaybı arasındaki ilişki
gender_churn = pd.crosstab(data['Gender'], data['Exited'], normalize='index') * 100
print(gender_churn)

gender_churn.plot(kind='bar', stacked=True)
plt.title('Cinsiyet ve Müşteri Kaybı')
plt.xlabel('Cinsiyet')
plt.ylabel('Yüzde')
plt.show()

# Ülke ve müşteri kaybı arasındaki ilişki
geo_churn = pd.crosstab(data['Geography'], data['Exited'], normalize='index') * 100
print(geo_churn)

geo_churn.plot(kind='bar', stacked=True)
plt.title('Ülke ve Müşteri Kaybı')
plt.xlabel('Ülke')
plt.ylabel('Yüzde')
plt.show()

# Yaşa göre müşteri kaybı dağılımı
# Thresholdlari budayalim 5,95
plt.figure(figsize=(10, 5))
sns.boxplot(x='Exited', y='Age', data=data)
plt.title('Yaş ve Müşteri Kaybı')
plt.show()

# Bakiye ve müşteri kaybı arasındaki ilişki
plt.figure(figsize=(10, 5))
sns.boxplot(x='Exited', y='Balance', data=data)
plt.title('Bakiye ve Müşteri Kaybı')
plt.show()

# Aldigim hata ( bazi soyadlarini korelasyon analizinde grafige eklemeye calistigi icin)
# sayisal sutunlari secme islemi yaptim.
numeric_columns = data.select_dtypes(include=['float64', 'int64'])

# Korelasyon matrisi
corr_matrix = numeric_columns.corr()
print(corr_matrix['Exited'].sort_values(ascending=False))

# Korelasyon matrisi grafiği

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korelasyon Matrisi')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Kullanılacak sütunları seçiyoruz (Sayısal ve kategorik sütunları düzenliyoruz)
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

# Kategorik değişkenleri one-hot encoding ile sayısal hale getirelim
X = pd.get_dummies(X, drop_first=True)

# Hedef değişken
y = data['Exited']

# cross validation yapalim!!







# Veriyi eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest modelini eğitelim
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Performans metriklerini ekrana yazdıralım
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:\n", conf_matrix)

# Test verisindeki churn olacak müşterileri işaretleyelim
X_test['Churn Tahmini'] = y_pred
churn_customers = X_test[X_test['Churn Tahmini'] == 1]

# Churn olacak müşterileri ekrana yazdıralım
print("\nChurn Olacak Müşteriler:")
print(churn_customers)

#Adım Adım Model Optimizasyonu

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Hiperparametre aralıklarını belirleyelim
param_grid = {
    'n_estimators': [100, 200, 300],      # Ağaç sayısı
    'max_depth': [None, 10, 20, 30],      # Maksimum derinlik
    'min_samples_split': [2, 5, 10],      # Bölünme için minimum örnek sayısı
    'min_samples_leaf': [1, 2, 4]         # Yaprak düğümlerindeki minimum örnek sayısı
}

# Random Forest modeli ve GridSearchCV ile hiperparametre araması
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2)

# Eğitim verisiyle en iyi parametreleri bulalım
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve puanı ekrana yazdıralım
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi Skor: {grid_search.best_score_}")


from sklearn.model_selection import cross_val_score

# 5 katlı çapraz doğrulama ile Random Forest modelinin performansını ölçelim
rf_best = grid_search.best_estimator_
cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5)

print(f"Çapraz Doğrulama Skorları: {cv_scores}")
print(f"Ortalama CV Skoru: {cv_scores.mean()}")



