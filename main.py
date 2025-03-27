import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

import joblib

# Import de jeu de données
df = pd.read_csv("data/listings.csv.gz", compression='gzip')
print(df.head())

# Selection de certaines colonnes
colonnes = [
    'price',
    'accommodates',
    'review_scores_rating',
    'reviews_per_month',
    'beds',
    'bathrooms_text',
    'has_availability'
]

df = df[colonnes]
print(df.head())

# Suppression des doublons
nb_doublons_initiaux = df.duplicated().sum()
print(f"Nombre de doublons avant suppression : {nb_doublons_initiaux}")

df = df.drop_duplicates()
nb_doublons_finaux = df.duplicated().sum()
print(f"Nombre de doublons après suppression : {nb_doublons_finaux}")

# Remplacer la valeur NaN des colonnes manquantes par la moyenne
colonnes_manquantes = ['reviews_per_month', 'review_scores_rating', 'beds']
df[colonnes_manquantes] = df[colonnes_manquantes].fillna(df[colonnes_manquantes].mean())

# Conversion des colonnes
df['price'] = df['price'].str.extract(r'(\d+)').astype(float)
df['bathrooms_text'] = df['bathrooms_text'].str.extract(r'(\d+)').fillna(0).astype(float)
print("\nAprès nettoyage :")
print(df.head())


# Statistiques générales
print("\n=== Statistiques descriptives ===")
print(df.describe().round(2))

# Vérification des types de données et valeurs manquantes
print("\n=== Info sur le DataFrame ===")
print(df.info())

# Vérification des valeurs uniques pour les variables catégorielles
print("\n=== Valeurs uniques pour 'has_availability' ===")
print(df['has_availability'].value_counts(dropna=False))

# 1. Distribution des Prix (Améliorée)
plt.figure(figsize=(9, 6))

# Histogramme + KDE
ax = sns.histplot(df['price'], bins=50, kde=True, color='royalblue', alpha=0.7)
plt.title('Distribution des Prix - Avant Nettoyage des Outliers', fontsize=16, pad=20)
plt.xlabel('Prix (€)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)

# Annotations
mean_price = df['price'].mean()
median_price = df['price'].median()
plt.axvline(mean_price, color='red', linestyle='--', linewidth=1.5, label=f'Moyenne: {mean_price:.2f}€')
plt.axvline(median_price, color='green', linestyle='--', linewidth=1.5, label=f'Médiane: {median_price:.2f}€')

# Zone interquartile
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
plt.axvspan(Q1, Q3, color='yellow', alpha=0.1, label='Zone interquartile')

plt.legend()
plt.grid(axis='y', alpha=0.3)
sns.despine()
plt.show()

# Matrice de Corrélation
df['has_availability'] = df['has_availability'].map({'t': 1, 'f': 0}).astype(float)

# Vérification finale des types
print("\n=== Types de données après conversion ===")
print(df.dtypes)

# Maintenant le calcul de corrélation devrait fonctionner
corr = df.corr()

# Masque pour le triangle supérieur
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(9, 5))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10})

plt.title('Matrice de Corrélation (Variables Numériques)', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Relation Prix vs Variables Clés
# Configuration des subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prix vs Capacité d'accueil
sns.scatterplot(ax=axes[0], x='accommodates', y='price', data=df, alpha=0.6, color='teal')
axes[0].set_title('Prix vs Capacité d\'Accueil', fontsize=14)
axes[0].set_xlabel('Nombre de Personnes', fontsize=12)
axes[0].set_ylabel('Prix (€)', fontsize=12)

# Prix vs Note moyenne
sns.regplot(ax=axes[1], x='review_scores_rating', y='price', data=df, 
            scatter_kws={'alpha':0.4, 'color':'orange'}, 
            line_kws={'color':'red', 'linewidth':2})
axes[1].set_title('Prix vs Note Moyenne', fontsize=14)
axes[1].set_xlabel('Note (sur 100)', fontsize=12)

# Prix vs Nombre de lits
sns.boxplot(ax=axes[2], x='beds', y='price', data=df[df['beds'] <= 5], palette='viridis')
axes[2].set_title('Prix par Nombre de Lits', fontsize=14)
axes[2].set_xlabel('Nombre de Lits', fontsize=12)

plt.tight_layout()
plt.show()

# Analyse de la Disponibilité
plt.figure(figsize=(9, 6))

# Boxplot avec hue pour une variable supplémentaire
sns.boxplot(x='has_availability', y='price', hue='accommodates', 
            data=df[df['accommodates'].between(2, 4)],
            palette='Set2', showfliers=False)

plt.title('Distribution des Prix par Disponibilité et Capacité', fontsize=16)
plt.xlabel('Disponibilité', fontsize=12)
plt.ylabel('Prix (€)', fontsize=12)
plt.xticks([0, 1], ['Non Disponible', 'Disponible'])
plt.legend(title='Capacité d\'Accueil', bbox_to_anchor=(1.05, 1), loc='upper left')

sns.despine()
plt.tight_layout()
plt.show()

# 1. Gestion des Outliers (Valeurs Aberrantes)
# Méthode IQR pour filtrer
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]

# Visualisation après nettoyage
plt.figure(figsize=(9, 6))
sns.histplot(df_clean['price'], bins=30, kde=True)
plt.title('Distribution des Prix après Suppression des Outliers')
plt.show()

# Suppression des outliers sur plusieurs colonnes
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

for col in ['accommodates', 'reviews_per_month', 'beds', 'bathrooms_text']:
    df_clean = remove_outliers(df_clean, col)

print(f"Nouvelle taille du dataset après suppression des outliers : {df_clean.shape}")

# Préparation des données pour la modélisation
X = df_clean.drop(columns=['price'])
y = df_clean['price']

# Encodage des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# Gestion des valeurs manquantes : Imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalisation des données
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
r2 = r2_score(y_test, y_pred)
print(f"Score R² du modèle : {r2:.4f}")

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(9, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Prix réel (€)")
plt.ylabel("Prix prédit (€)")
plt.title("Prix Réel vs Prix Prédit")
plt.show()

# Calculer l'erreur moyenne absolue (MAE) et l'erreur quadratique moyenne (RMSE)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Sauvegarder le modèle pour le réutiliser avec joblib
joblib.dump(model, "model_airbnb.pkl")
model = joblib.load("model_airbnb.pkl")




