import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, fbeta_score
import joblib
import io
import base64

class PythiaSystem:
    def __init__(self):
        self.regression_models = {
            'SVM': SVR(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Ridge': Ridge(),
            'AdaBoost': AdaBoostRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'KNN': KNeighborsRegressor(),
            'Neural Network': MLPRegressor()
        }

        self.classification_models = {
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'AdaBoost': AdaBoostClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'KNN': KNeighborsClassifier(),
            'Neural Network': MLPClassifier()
        }

        self.ues = ['Moyenne Générale']
        
        # Méthode pour tracer la courbe ROC AUC


    def plot_roc_auc(self, y_true, y_pred, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
    
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
    
        # Convertir le graphique en image en base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        # Convertir la matrice de confusion en image en base64
        buf = io.BytesIO()
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Matrice de confusion - {model_name}')
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"

    def load_data(self):
        data = pd.read_csv('L1-GL-2020-2021.csv', encoding='utf-8', skiprows=2)
        data = data[data['Matricule'].notna()]  # Supprime les lignes vides
        return data

    def preprocess_data(self, data):
        features = ['Crédit SEM 1', 'Crédit SEM 2', 'Total Crédit', 'Moy 1er sem', 'Moy 2eme sem']
        X = data[features]
        y = data[['Moyenne Générale']]
        
        # Remplacer les NaN dans les données par la moyenne de chaque colonne (ou une autre stratégie)
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y
    

    def evaluate_regression(self, X, y):
        results = {}
        for ue in self.ues:
            ue_results = {}
            for name, model in self.regression_models.items():
                scores = cross_val_score(model, X, y[ue], cv=5, scoring='neg_mean_squared_error')
                rmse = np.sqrt(-scores.mean())
                ue_results[name] = rmse
            results[ue] = ue_results
        return results

    def evaluate_classification(self, X, y):
        results = {}
        f2_scorer = make_scorer(fbeta_score, beta=2)
        for ue in self.ues:
            ue_results = {}
            y_class = (y[ue] >= 10).astype(int)  # Seuil de validation à 10
            for name, model in self.classification_models.items():
                scores = cross_val_score(model, X, y_class, cv=2, scoring=f2_scorer)
                ue_results[name] = scores.mean()

                # Entraîner et prédire pour visualiser les métriques
                model.fit(X, y_class)
                y_pred = model.predict(X)

                # Affichage des courbes ROC AUC et matrices de confusion
                self.plot_roc_auc(y_class, y_pred, name)  # Correction ici avec self.
                self.plot_confusion_matrix(y_class, y_pred, name)  # Correction ici aussi.
                
            results[ue] = ue_results
        return results


    def train_best_models(self, X, y):
        best_models = {}
        for ue in self.ues:
            reg_model = min(self.regression_models.items(), key=lambda x: self.evaluate_regression(X, y)[ue][x[0]])
            clf_model = max(self.classification_models.items(), key=lambda x: self.evaluate_classification(X, y)[ue][x[0]])

            reg_model[1].fit(X, y[ue])
            y_class = (y[ue] >= 10).astype(int)
            clf_model[1].fit(X, y_class)

            best_models[ue] = {
                'regression': reg_model,
                'classification': clf_model
            }
        return best_models

    def predict(self, best_models, X):
       predictions = []  # Liste pour stocker les prédictions pour chaque étudiant
       for ue, models in best_models.items():
           reg_pred = models['regression'][1].predict(X)  # Prédictions de régression
           clf_pred = models['classification'][1].predict(X)  # Prédictions de classification
           
           # Combinez les prédictions en une liste de dictionnaires
           for i in range(len(X)):
               predictions.append({
                   'regression': reg_pred[i],  # Prédiction de régression pour l'étudiant i
                   'classification': clf_pred[i],  # Prédiction de classification pour l'étudiant i
                   'ue': ue  # Unité d'enseignement
               })
       return predictions

    def run(self):
        data = self.load_data()
        X, y = self.preprocess_data(data)

        print("Évaluation des modèles de régression:")
        reg_results = self.evaluate_regression(X, y)
        for ue, models in reg_results.items():
            print(f"\n{ue}:")
            for model, rmse in models.items():
                print(f"{model}: RMSE = {rmse:.4f}")

        print("\nÉvaluation des modèles de classification:")
        clf_results = self.evaluate_classification(X, y)
        for ue, models in clf_results.items():
            print(f"\n{ue}:")
            for model, f1 in models.items():
                print(f"{model}: F1-score = {f1:.4f}")

        best_models = self.train_best_models(X, y)

        print("\nMeilleurs modèles:")
        for ue, models in best_models.items():
            print(f"\n{ue}:")
            print(f"Régression: {models['regression'][0]}")
            print(f"Classification: {models['classification'][0]}")

        # Prédictions pour les 5 premiers étudiants
        X_sample = X
        predictions = self.predict(best_models, X_sample)

        print("\nPrédictions pour les 5 premiers étudiants:")
        for ue, preds in predictions.items():
            print(f"\n{ue}:")
            print("Régression (Moyenne prédite):")
            print(preds['regression'])
            print("Classification (Validation prédite, 1=Validé, 0=Non validé):")
            print(preds['classification'])


      