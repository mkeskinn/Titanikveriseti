#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Programlama Dilleri Final Ödevi


# In[2]:


#1.Soru:
#Veriseti, Titanik gemisine bilet alan yolcuların yaş, cinsiyet, bilet fiyatları vs. bilgilerini içermektedir. 


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


#2.Soru
dataset=pd.read_csv("veriseti.csv")
dataset


# In[5]:


#3.Soru:
dataset.head(10)


# In[6]:


#4.Soru:
dataset.shape


# In[7]:


#5.Soru:
survived_sutunu = dataset['Survived']
sayilar = survived_sutunu.value_counts()
#Histogram oluşturma
sayilar.plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('Survived Sutünu Değerleri')
plt.ylabel('Frekans')
plt.title('Survived Sutünu Histogramı')
plt.show()


# In[33]:


#6.Soru:
veri_turleri = dataset.dtypes

print(veri_turleri)


# In[35]:


#7.Soru:
istatistikler = dataset.describe()

print(istatistikler)


# In[36]:


#8.Soru:
dataset.isnull().sum()


# In[42]:


#9.Soru:
# Eksik veri içeren satırları sil
dataset_temiz = dataset.dropna()


# In[46]:


#10.Soru:
ortalama_age = dataset['Age'].mean()
print(f"Age kolonunun ortalaması: {ortalama_age}")


# In[47]:


#11.Soru:
en_yuksek_deger = dataset['Ticket'].max()
print(f"Ticket kolonundaki en yüksek değer: {en_yuksek_deger}")


# In[57]:


#12.Soru:
ortalama_fare = dataset['Fare'].mean()
print(f"Fare kolonunun ortalaması: {ortalama_fare}")


# In[58]:


#13.Soru
farkli_veri_sayisi = dataset['Age'].nunique()
print(f"Age kolonundaki farklı veri sayısı: {farkli_veri_sayisi}")


# In[60]:


#14.Soru:
sayisal_veriler = dataset.select_dtypes(include=[np.number])
korelasyon_matrisi = sayisal_veriler.corr()
print("Korelasyon Matrisi:")
print(korelasyon_matrisi)


# In[61]:


#15.Soru
#Bağımsız değişkenler=(x)
x = dataset.drop('Survived', axis=1)  # 'Survived' sütununu çıkartarak bağımsız değişkenleri oluştur

#Bağımlı değişken=(y)
y = dataset['Survived']  # 'Survived' sütunu bağımlı değişken

# x ve y'yi görüntüle
print("Bağımsız Değişkenler (x):")
print(x.head())

print("\nBağımlı Değişken (y):")
print(y.head())


# In[41]:


#16.Soru:
from sklearn.model_selection import train_test_split
import pandas as pd
# Bağımsız değişkenler (x) ve bağımlı değişken (y) tanımlama
x = dataset.drop('Survived', axis=1)
y = dataset['Survived']

# Veriyi test ve eğitim olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ayırılmış veriyi gösterme
print("Eğitim (Train) Verisi:")
print(x_train.head())
print(y_train.head())

print("\nTest Verisi:")
print(x_test.head())
print(y_test.head())


# In[67]:


#17.Soru Z-Skor
from sklearn.preprocessing import StandardScaler
import pandas as pd
scaler = StandardScaler()
numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
print(dataset.head())


# In[14]:


#18.Soru: #"Açıklaması bir sonraki satırdadır."
#Gerekli kütüphaneleri ekleyin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Veri setini yükleyin
dataset = pd.read_csv("veriseti.csv")

# Eksik değer içeren satırları sil
dataset.dropna(inplace=True)

# Özellikleri ve hedef değişkeni belirleyin
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

# Kategorik değişkenleri sayısala dönüştürün
dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayırın
X = dataset[features]
y = dataset[target]

# Eğitim ve test veri setlerini oluşturun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest sınıflandırma modelini oluşturun
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yapın
y_pred = rf_model.predict(X_test)

# Model performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Modelin tahminleri ve gerçek değerleri görüntüle
print("Tahminler:\n", y_pred)
print("Gerçek Değerler:\n", y_test)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)


# In[ ]:


#18.soru Açıklaması:
#Verisetimde random forest kullandım. Bu sayede modelin performansını değerlendirdim. 


# In[10]:


#19.Soru:
from sklearn.metrics import confusion_matrix

# Confusion matrixi oluşturun
cm = confusion_matrix(y_test, y_pred)

# Confusion matrixi ekrana yazdırın
print("Confusion Matrix:")
print(cm)


# In[11]:


#20.Soru #"Açıklamalar bir sonraki satırda"
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Modelin accuracy değerini bulun
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Modelin sensitivity (recall) değerini bulun
sensitivity = recall_score(y_test, y_pred)
print(f"Sensitivity (Recall): {sensitivity}")

# Modelin F1 score değerini bulun
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")


# In[ ]:


#20. Soru Açıklama
#Accuracy, bir sınıflandırma modelinin doğru tahminlerin toplam veri sayısına oranını ifade eder. 
#Yüksek bir accuracy, genel performansın iyi olduğunu gösterir. Ancak, dengesiz veri setlerinde yanıltıcı olabilir,
#çünkü büyük sınıflara odaklanabilir ve küçük sınıfları ihmal edebilir.
#Bu nedenle, özellikle dengesiz veri setlerinde, diğer metriklere de bakmak önemlidir.

