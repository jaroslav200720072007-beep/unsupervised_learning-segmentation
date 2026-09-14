# 🧠 Customer Segmentation — Навчання без вчителя (Unsupervised Learning)

Проєкт з аналізу та сегментації клієнтів на основі демографічних даних та історії покупок. Використано методи **EDA**, **feature engineering** та три алгоритми кластеризації: **K-Means**, **Mean Shift** та **DBSCAN**, з візуалізацією результатів через **PCA**.

---

## 📌 Зміст

- [Про проєкт](#-про-проєкт)
- [Датасет](#-датасет)
- [Технології](#-технології)
- [Етапи роботи](#-етапи-роботи)
  - [1. Завантаження та очищення даних](#1-завантаження-та-очищення-даних)
  - [2. Розвідувальний аналіз (EDA)](#2-розвідувальний-аналіз-eda)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Масштабування даних](#4-масштабування-даних)
  - [5. Кластеризація](#5-кластеризація)
  - [6. Зниження розмірності та візуалізація кластерів (PCA)](#6-зниження-розмірності-та-візуалізація-кластерів-pca)
- [Результати](#-результати)
- [Як запустити проєкт](#-як-запустити-проєкт)
- [Автор](#-автор)

---

## 📖 Про проєкт

Мета проєкту — виявити приховані сегменти (групи) клієнтів на основі їхнього віку, доходу, витрат та поведінки покупок, без використання заздалегідь відомих міток класів. Такий підхід дозволяє бізнесу краще розуміти свою аудиторію та будувати персоналізовані маркетингові стратегії.

---

## 📊 Датасет

Використано датасет з інформацією про клієнтів, що містить **2240 записів** та **29 колонок**, зокрема:

- Демографічні дані: `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`
- Дані про покупки: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- Поведінкові дані: `NumWebPurchases`, `NumStorePurchases`, `NumCatalogPurchases`, `NumWebVisitsMonth`
- Дані про участь у маркетингових кампаніях: `AcceptedCmp1-5`, `Response`

---

## 🛠 Технології

- **Python 3**
- **Pandas**, **NumPy** — обробка та аналіз даних
- **Matplotlib**, **Seaborn** — візуалізація
- **Scikit-learn** — масштабування, кластеризація (`KMeans`, `MeanShift`, `DBSCAN`), зниження розмірності (`PCA`)

---

## 🔍 Етапи роботи

### 1. Завантаження та очищення даних

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(full_path)
df.head()
```

Перевірка структури даних:

```python
df.columns
df.info()
```

Пошук та видалення пропущених значень (24 пропуски у колонці `Income`):

```python
df.isna().sum().sum()   # 24
df = df.dropna()
df.isna().sum().sum()   # 0
```

Загальна статистика та категоріальні розподіли:

```python
df.describe()
df['Education'].value_counts()
df["Marital_Status"].value_counts()
```

Приведення дати реєстрації клієнта до правильного формату:

```python
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
```

---

### 2. Розвідувальний аналіз (EDA)

**Розподіл віку клієнтів:**

```python
sns.histplot(df["Age"], bins=30, kde=True)
```

![Розподіл віку](<img width="710" height="460" alt="Снимок экрана — 2026-09-14 в 17 07 21" src="https://github.com/user-attachments/assets/47bf29bc-db1b-464b-be85-a76b7f327627" />
)


**Загальні витрати клієнтів:**

```python
sns.histplot(df['Total_Spending'], bins=30, kde=True)
```

![Загальні витрати](<img width="720" height="471" alt="Снимок экрана — 2026-09-14 в 17 06 42" src="https://github.com/user-attachments/assets/56399f6e-b976-4e22-a2bd-9cb940031fc5" />
)

**Витрати залежно від сімейного стану:**

```python
sns.boxplot(x="Marital_Status", y="Total_Spending", data=df)
```

![Витрати vs Сімейний стан](<img width="658" height="448" alt="Снимок экрана — 2026-09-14 в 17 05 32" src="https://github.com/user-attachments/assets/d0591387-a740-433e-946d-05127de98161" />
)

**Кореляційна матриця ключових ознак:**

```python
corr = df[["Income", "Age", "Recency", "Total_Spending",
           "NumWebPurchases", "NumStorePurchases"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
```

![Кореляційна матриця](<img width="703" height="574" alt="Снимок экрана — 2026-09-14 в 17 04 56" src="https://github.com/user-attachments/assets/226263e6-2617-4635-8380-353a1a4cb08c" />)

**Середні витрати за рівнем освіти:**

```python
group1 = df.groupby("Education")["Total_Spending"].mean().sort_values(ascending=False)
group1.plot(kind="bar", color="skyblue")
```

![Витрати за освітою](<img width="606" height="517" alt="Снимок экрана — 2026-09-14 в 17 04 29" src="https://github.com/user-attachments/assets/6c44d583-54fd-4576-ab7d-c07623460e55" />
)

**Середній дохід за віковими групами:**

```python
bins = [18, 30, 40, 50, 60, 70, 90]
labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

group3 = df.groupby("AgeGroup")["Income"].mean()
group3.plot(kind="barh", color="green")
```

![Дохід за віковими групами](<img width="618" height="413" alt="Снимок экрана — 2026-09-14 в 17 03 59" src="https://github.com/user-attachments/assets/78fdd0e6-3d84-4a52-bc0d-026791ed82c3" />
)

---

### 3. Feature Engineering

```python
df["Age"] = 2025 - df["Year_Birth"]

df["Total_Children"] = df['Kidhome'] + df['Teenhome']

spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
              'MntFishProducts', 'MntSweetProducts']
df['Total_Spending'] = df[spend_cols].sum(axis=1)

df["Customer_Since"] = (pd.Timestamp("today") - df["Dt_Customer"])
df['Customer_Since'] = df['Customer_Since'].dt.days
```

Нові ознаки, отримані на цьому кроці:

| Нова ознака | Опис |
|---|---|
| `Age` | Вік клієнта (2025 − Year_Birth) |
| `Total_Children` | Сума `Kidhome` + `Teenhome` |
| `Total_Spending` | Сума витрат за всіма категоріями товарів |
| `Customer_Since` | Кількість днів з моменту реєстрації клієнта |
| `AgeGroup` | Віковий діапазон клієнта (18-29, 30-39 ... 70+) |

---

### 4. Масштабування даних

```python
features = ["Age", "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases"]
X = df[features].copy()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Ознаки стандартизовано (середнє = 0, дисперсія = 1), щоб уникнути домінування ознак з великим масштабом значень (наприклад, `Income` над `NumStorePurchases`).

---

### 5. Кластеризація

#### 🔹 K-Means

Оптимальну кількість кластерів визначено методом ліктя (**Elbow method**):

```python
from sklearn.cluster import KMeans

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
```

![Метод ліктя](<img width="692" height="420" alt="Снимок экрана — 2026-09-14 в 17 02 55" src="https://github.com/user-attachments/assets/24206304-3e59-4411-956b-c0e7f89c9610" />
)

Найкраща кількість кластерів — **6**:

```python
km = KMeans(n_clusters=6)
km.fit(X_scaled)
df['Cluster_km'] = km.labels_

cluster_summary = df.groupby("Cluster_km")[features].mean()
df["Cluster_km"].value_counts()
```

#### 🔹 Mean Shift

Кластеризація без заданої кількості кластерів — алгоритм сам визначає їхню кількість на основі щільності точок.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth

bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_scaled)
df['Cluster_MeanShift'] = ms.labels_

df['Cluster_MeanShift'].value_counts()
cluster_summary_ms = df.groupby('Cluster_MeanShift')[features].mean()
```

#### 🔹 DBSCAN

Кластеризація на основі щільності, що дозволяє виявляти шумові точки (outliers) окремо від кластерів.

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.8, min_samples=5)
df['Cluster_DBSCAN'] = db.fit_predict(X_scaled)

df['Cluster_DBSCAN'].value_counts()
```

---

### 6. Зниження розмірності та візуалізація кластерів (PCA)

Для наочної візуалізації багатовимірних даних застосовано зменшення розмірності до 2 компонент (**PCA**).

**Кластери K-Means:**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster_km", data=df)
```

![K-Means кластери (PCA)](<img width="590" height="431" alt="Снимок экрана — 2026-09-14 в 17 02 07" src="https://github.com/user-attachments/assets/9a784189-b4d5-4c9f-b1b9-97deb7f0919d" />
)

**Кластери Mean Shift:**

```python
pca = PCA(n_components=2)
data_2d = pca.fit_transform(X_scaled)

df['pca1'] = data_2d[:, 0]
df['pca2'] = data_2d[:, 1]

sns.scatterplot(x=df['pca1'], y=df['pca2'], hue='Cluster_MeanShift', data=df)
```

![Mean Shift кластери (PCA)](<img width="598" height="423" alt="Снимок экрана — 2026-09-14 в 17 01 12" src="https://github.com/user-attachments/assets/f41b8109-ec6e-4705-a73e-770cce9aa76b" />)

**Кластери DBSCAN:**

```python
pca = PCA(n_components=2)
data_db = pca.fit_transform(X_scaled)

df['col1db'] = data_db[:, 0]
df['col2db'] = data_db[:, 1]

sns.scatterplot(x='col1db', y='col2db', hue='Cluster_DBSCAN', data=df, palette='Set1')
```

<img width="622" height="437" alt="Снимок экрана — 2026-09-14 в 17 00 19" src="https://github.com/user-attachments/assets/e4a29818-da5f-4cc4-977b-f8067dccdb59" />


---

## 📈 Результати

- Виявлено **6 сегментів клієнтів** методом K-Means з різними профілями доходу, витрат та активності покупок.
- Порівняно три підходи до кластеризації: **K-Means** (чіткі сферичні кластери), **Mean Shift** (автоматичний підбір кількості кластерів) та **DBSCAN** (виявлення шумових точок/аномалій).
- Побудовано профілі кожного кластера (`cluster_summary`) для подальшого бізнес-аналізу та таргетування маркетингових кампаній.


## 👤 Автор

[LinkedIn]([https://linkedin.com/in/ВАШ_ПРОФІЛЬ](https://www.linkedin.com/in/%D1%8F%D1%80%D0%BE%D1%81%D0%BB%D0%B0%D0%B2-%D1%81%D1%82%D0%B5%D1%86%D1%8E%D0%BA-0943723a1))

---
