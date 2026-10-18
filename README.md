# Lab 1, Partie 2 : Classification multiclasse (Predictive Maintenance)

**Objectif :** prendre en main PyTorch pour résoudre une tâche de classification multiclasse avec un réseau de type MLP.
**Problème :** prédire le type de panne (`Failure Type`) d'une machine à partir de 6 variables : type de machine, températures de l'air et du process, vitesse de rotation, couple et usure de l'outil.

**Dataset :** [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (10 000 lignes).
**Outils :** Kaggle, PyTorch, scikit-learn, skorch, imbalanced-learn, pandas, matplotlib, seaborn.
Les graphes et les sorties détaillées sont dans le notebook `part2_classification_maintenance.ipynb`.

---

## 1. Pré-traitement

- Aucune valeur manquante, aucun doublon.
- J'ai retiré **27 lignes** aux étiquettes contradictoires : 9 avec `Target = 1` mais « No Failure », et 18 « Random Failures » avec `Target = 0`. Il reste **9 973 lignes et 5 classes**.
- J'ai supprimé `UDI` et `Product ID` (simples identifiants) et `Target`, qui est déduite de la classe à prédire (fuite d'information).
- `Type` est encodé de façon ordinale (L = 0, M = 1, H = 2) et la cible est transformée en entiers (0 à 4).
- Split 80 % / 20 % **stratifié** (7 978 lignes en train, 1 995 en test), puis `StandardScaler` ajusté **uniquement sur le train** pour éviter la fuite de données.

## 2. Analyse exploratoire (EDA)

Les données sont **très déséquilibrées** :

| Classe | Effectif |
|---|---|
| No Failure | 9 643 (96,7 %) |
| Heat Dissipation Failure | 112 |
| Power Failure | 95 |
| Overstrain Failure | 78 |
| Tool Wear Failure | 45 |

Un modèle qui répondrait toujours « No Failure » aurait donc près de 97 % d'accuracy sans rien détecter. Il faut suivre le F1 macro et la sensibilité par classe.

Les variables ont des formes variées : `torque` est en cloche autour de 40 Nm, `rot_speed` est asymétrique à droite, `tool_wear` est quasi uniforme de 0 à 200 minutes. Chaque panne a un profil de mesures distinct :
- **Heat Dissipation** : température de l'air plus élevée (302,6 K) et vitesse basse (1 338 rpm).
- **Overstrain** : forte usure (208 min) et couple élevé (56,9 Nm).
- **Tool Wear** : usure maximale (216,6 min).
- **Power Failure** : vitesse élevée (1 764 rpm) et couple supérieur à la normale.

`Type` a une influence limitée sur les pannes. Le lien entre mesures et classes est non linéaire, ce qui justifie un réseau de neurones.

## 3. Data augmentation (SMOTE)

J'ai appliqué **SMOTE** pour que les 5 classes aient le même effectif dans le train. Il est appliqué **après le split et uniquement sur le train** : le test reste réel et déséquilibré, comme en situation de production.

## 4. Architecture du réseau

MLP avec 6 entrées, des couches cachées `Linear + ReLU` et **5 sorties** (logits, sans activation). La perte est `CrossEntropyLoss`, qui applique elle-même le softmax. La classe prédite est l'`argmax` des sorties.

## 5. GridSearch

`GridSearchCV` avec `skorch`, validation croisée stratifiée à 3 plis. SMOTE est placé dans un **pipeline** : il n'est appliqué que sur la partie entraînement de chaque pli. Critère de sélection : **F1 macro** (l'accuracy est trompeuse ici).

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD, RMSprop |
| Epochs | 20, 40 |
| Architecture | (32), (64, 32), (128, 64, 32) |

**Meilleure configuration :** lr = [...], optimiseur = [...], epochs = [...], architecture = [...], F1 macro CV = [...].

## 6. Courbes Loss/Epochs et Accuracy/Epochs

La loss d'entraînement descend de 0,17 à environ 0,02 et l'accuracy train atteint presque 99,5 %. La loss de test se stabilise autour de 0,15 à 0,22 avec des oscillations, et l'accuracy test plafonne entre 95 et 97 %. Cet écart indique un **début d'overfitting** : le modèle apprend très bien le train, y compris les points synthétiques de SMOTE. Il est amplifié par le fait que le train est équilibré alors que le test est réel et déséquilibré, donc les deux courbes ne mesurent pas exactement la même chose.

## 7. Métriques

| | Accuracy | Précision (macro) | Sensibilité (macro) | F1 (macro) |
|---|---|---|---|---|
| Train (réel) | [...] | [...] | [...] | [...] |
| Test | [...] | [...] | [...] | [...] |

Classe la mieux détectée : [...] (sensibilité [...]). Classe la plus difficile : [...] (sensibilité [...]). Les classes de panne ont très peu d'exemples dans le test, donc ces scores sont peu stables : une seule erreur change beaucoup le résultat.

## 8. Régularisation

J'ai comparé le modèle de base à cinq variantes (L2 avec weight decay 1e-4, Dropout 0,2, BatchNorm, Early Stopping, et une combinaison), avec les mêmes hyperparamètres et la même initialisation.

| Modèle | Test Accuracy | Test Sensibilité | Test F1 macro | Epochs |
|---|---|---|---|---|
| 1-Base | [...] | [...] | [...] | [...] |
| 2-L2 | [...] | [...] | [...] | [...] |
| 3-Dropout | [...] | [...] | [...] | [...] |
| 4-BatchNorm | [...] | [...] | [...] | [...] |
| 5-EarlyStop | [...] | [...] | [...] | [...] |
| 6-Combiné | [...] | [...] | [...] | [...] |

Sur les courbes de test, toutes les variantes convergent vers une loss de 0,15 à 0,22. La **BatchNorm** converge le plus vite avec la loss de test la plus basse, tandis que **Dropout** et la **combinaison** ralentissent l'apprentissage. Au final : [conclusion à adapter selon le F1 macro du tableau].

## 9. Synthèse personnelle

Dans cette partie, j'ai appris à traiter un problème multiclasse déséquilibré de bout en bout : nettoyage, standardisation, EDA, SMOTE, MLP avec `CrossEntropyLoss`, GridSearch et régularisation.

- Avec 96,7 % de « No Failure », l'accuracy est trompeuse : il faut suivre le F1 macro, la sensibilité et la matrice de confusion.
- `Target` doit être supprimée (fuite de données).
- SMOTE s'applique après le split, uniquement sur le train, et dans un pipeline pendant la validation croisée.
- Le train équilibré et le test réel ne mesurent pas la même chose : l'écart de leurs courbes est à interpréter avec prudence.

**Difficultés :** le très faible nombre d'exemples de certaines pannes (45 pour Tool Wear), le temps du GridSearch et la perte de session sur Kaggle.
**Pistes d'amélioration :** variables dérivées (écart de température, puissance = couple × vitesse), poids de classes dans la loss, SMOTENC pour `Type`, jeu de validation séparé pour l'early stopping, comparaison avec Random Forest ou XGBoost.

## Exécution

```bash
pip install torch skorch scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

Ouvrir le notebook sur Kaggle (en ajoutant le dataset `shivamb/machine-predictive-maintenance-classification`) ou sur Google Colab, puis exécuter toutes les cellules.
