# Lab 1, Partie 2 : Classification multiclasse (Predictive Maintenance)

**Objectif :** prendre en main PyTorch pour la classification multiclasse avec un réseau de type MLP.
**Problème :** prédire le type de panne (`Failure Type`) d'une machine à partir de 6 variables : type de machine, température de l'air, température du process, vitesse de rotation, couple et usure de l'outil.

**Dataset :** [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (10 000 lignes).
**Outils :** Kaggle (CPU), PyTorch, scikit-learn, skorch, imbalanced-learn, pandas, matplotlib, seaborn.
Les graphes et les sorties détaillées sont dans le notebook `lab1-part2-predictive-maintenance.ipynb`.

---

## 1. Pré-traitement

- Aucune valeur manquante, aucun doublon.
- J'ai retiré **27 lignes** aux étiquettes contradictoires : 9 avec `Target = 1` mais « No Failure », et 18 « Random Failures » avec `Target = 0`. Il reste **9 973 lignes et 5 classes**.
- J'ai supprimé `UDI` et `Product ID` (simples identifiants) et `Target`, qui est déduite de la classe à prédire (fuite d'information).
- `Type` est encodé de façon ordinale (L = 0, M = 1, H = 2) et la cible est transformée en entiers (0 à 4).
- Split 80 % / 20 % **stratifié** : 7 978 lignes en train, 1 995 en test.
- `StandardScaler` ajusté **uniquement sur le train** pour éviter la fuite de données.

## 2. Analyse exploratoire (EDA)

Les données sont **très déséquilibrées** :

| Classe | Effectif | Part |
|---|---|---|
| No Failure | 9 643 | 96,69 % |
| Heat Dissipation Failure | 112 | 1,12 % |
| Power Failure | 95 | 0,95 % |
| Overstrain Failure | 78 | 0,78 % |
| Tool Wear Failure | 45 | 0,45 % |

Un modèle qui répondrait toujours « No Failure » aurait près de 97 % d'accuracy sans rien détecter. Il faut donc suivre le F1 macro et la sensibilité par classe.

Les variables ont des formes variées : `torque` est en cloche autour de 40 Nm, `rot_speed` est asymétrique à droite (jusqu'à environ 2 900 rpm), `tool_wear` est quasi uniforme de 0 à 200 minutes. Les échelles sont très différentes, donc la standardisation est nécessaire.

Chaque panne a un profil de mesures distinct (moyennes par classe) :

| Panne | Profil |
|---|---|
| Heat Dissipation | air 302,6 K (contre 300,0), vitesse basse (1 338 rpm) |
| Overstrain | usure 208 min, couple 56,9 Nm |
| Tool Wear | usure maximale (216,6 min) |
| Power Failure | vitesse élevée (1 764 rpm), couple 48,5 Nm |

`Type` influence peu les pannes (96 à 98 % de « No Failure » pour chaque type). Les mesures permettent de séparer les classes, mais de façon non linéaire, ce qui justifie un réseau de neurones.

## 3. Data augmentation (SMOTE)

J'ai appliqué **SMOTE** sur le train, **après le split**, pour que les 5 classes aient le même effectif. Le test reste réel et déséquilibré, comme en situation de production.

| Classe | Avant SMOTE | Après SMOTE |
|---|---|---|
| No Failure | 7 714 | 7 714 |
| Heat Dissipation | 90 | 7 714 |
| Power Failure | 76 | 7 714 |
| Overstrain | 62 | 7 714 |
| Tool Wear | 36 | 7 714 |

Le train passe ainsi à 38 570 lignes. La classe Tool Wear, qui part de seulement 36 exemples réels, est fortement synthétique après SMOTE.

## 4. Architecture du réseau

MLP avec 6 entrées, des couches cachées `Linear + ReLU` et **5 sorties** (logits, sans activation). La perte est `CrossEntropyLoss`, qui applique elle-même le softmax. La classe prédite est l'`argmax` des sorties. L'entraînement utilise une boucle PyTorch manuelle (`zero_grad`, forward, `backward`, `step`).

## 5. GridSearch

`GridSearchCV` avec `skorch`, validation croisée stratifiée à 3 plis, soit 36 combinaisons et 108 entraînements. SMOTE est placé dans un **pipeline** : il n'est appliqué que sur la partie entraînement de chaque pli. Critère : **F1 macro**.

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD, RMSprop |
| Epochs | 20, 40 |
| Architecture | (32), (64, 32), (128, 64, 32) |

**Meilleure configuration :** lr = 0.001, optimiseur = Adam, epochs = 40, architecture = (128, 64, 32), **F1 macro CV = 0,419**.

Les 10 meilleures configurations utilisent toutes 40 epochs. Adam domine le classement, suivi de RMSprop, et SGD est moins performant. Le F1 macro en validation croisée est bas, car chaque pli ne contient que quelques exemples réels des classes rares.

## 6. Courbes Loss/Epochs et Accuracy/Epochs

Pour le modèle 1, la loss d'entraînement descend de 0,17 à environ 0,02 et l'accuracy train atteint **99,54 %**. La loss de test chute vite (0,45 à 0,15 environ) puis se stabilise entre 0,15 et 0,22 avec des oscillations. L'accuracy test plafonne à **95,74 %**.

Cet écart indique un **début d'overfitting** : le modèle apprend très bien le train, y compris les points synthétiques de SMOTE. Il est amplifié par le fait que le train est équilibré alors que le test est réel et déséquilibré, donc les deux courbes ne mesurent pas exactement la même chose.

## 7. Métriques (modèle 1)

| | Accuracy | Précision (macro) | Sensibilité (macro) | F1 (macro) |
|---|---|---|---|---|
| Train (réel, non augmenté) | 0,9777 | 0,7524 | 0,9954 | 0,8183 |
| Test | 0,9574 | 0,5887 | 0,8066 | 0,6596 |

Détail par classe sur le test :

| Classe | Précision | Sensibilité | F1 | Effectif test |
|---|---|---|---|---|
| Heat Dissipation | 0,61 | 0,91 | 0,73 | 22 |
| No Failure | 0,99 | 0,96 | 0,98 | 1 929 |
| Overstrain | 0,67 | 0,88 | 0,76 | 16 |
| Power Failure | 0,59 | 0,84 | 0,70 | 19 |
| Tool Wear | 0,08 | 0,44 | 0,14 | 9 |

**Interprétation :**
- L'accuracy (95,7 %) est trompeuse : le F1 macro n'est que de 0,66.
- Le modèle **détecte bien les pannes** (sensibilité de 0,84 à 0,91 pour trois classes), mais avec beaucoup de **fausses alertes** (précision de 0,59 à 0,67). C'est l'effet de SMOTE : le réseau a appris à prédire « panne » plus souvent que dans la réalité.
- **Tool Wear est la classe la plus difficile** (F1 = 0,14, précision 0,08). Elle n'a que 36 exemples réels en train, 9 en test, et le reste est synthétique.
- Avec si peu d'exemples en test, une seule erreur change beaucoup les scores : ils sont peu stables.

## 8. Régularisation

J'ai comparé le modèle de base à cinq variantes (L2 avec weight decay 1e-4, Dropout 0,2, BatchNorm, Early Stopping avec patience 5, et une combinaison L2 + Dropout + BatchNorm), avec les mêmes hyperparamètres et la même initialisation.

| Modèle | Test Accuracy | Test Précision | Test Sensibilité | Test F1 macro | Epochs |
|---|---|---|---|---|---|
| 1-Base | 0,9574 | 0,5887 | 0,8066 | 0,6596 | 40 |
| 2-L2 | 0,9539 | 0,5859 | 0,8342 | 0,6639 | 40 |
| 3-Dropout | 0,9534 | 0,5981 | **0,8616** | 0,6774 | 40 |
| 4-BatchNorm | **0,9639** | **0,6361** | 0,7885 | **0,6838** | 40 |
| 5-EarlyStop | 0,9559 | 0,5778 | 0,7973 | 0,6493 | 18 |
| 6-Combiné | 0,9479 | 0,5724 | 0,8447 | 0,6524 | 40 |

**Analyse :**
- La **BatchNorm** obtient le meilleur F1 macro (0,684 contre 0,660 pour le modèle de base), ainsi que la meilleure accuracy et la meilleure précision. Sur les courbes, elle converge aussi le plus vite avec la loss de test la plus basse.
- Le **Dropout** est deuxième (0,677) et a la meilleure sensibilité (0,862) : il détecte plus de pannes, au prix de plus de fausses alertes.
- L2 apporte un gain très faible. L'Early Stopping (arrêt à l'epoch 18) et la combinaison font légèrement moins bien que le modèle de base.
- **Les écarts restent petits** (au maximum 0,03 de F1 macro). Le test ne contient que 66 exemples de panne, donc ces différences ne sont pas forcément significatives.

**Conclusion :** la régularisation améliore un peu la généralisation (surtout BatchNorm et Dropout), mais pas de façon spectaculaire. Le surapprentissage vient surtout des points synthétiques de SMOTE, qu'une régularisation classique corrige peu.

*Limite :* l'Early Stopping surveille la loss de test, faute de jeu de validation séparé. Idéalement, il faudrait un troisième ensemble.

## 9. Synthèse personnelle

Dans cette partie, j'ai appris à traiter un problème multiclasse déséquilibré de bout en bout : nettoyage, standardisation, EDA, SMOTE, MLP avec `CrossEntropyLoss`, GridSearch et régularisation.

- Avec 96,7 % de « No Failure », l'accuracy est trompeuse : il faut suivre le F1 macro, la sensibilité par classe et la matrice de confusion.
- `Target` doit être supprimée (fuite de données), et les lignes contradictoires retirées.
- SMOTE s'applique après le split, uniquement sur le train, et dans un pipeline pendant la validation croisée.
- Le train équilibré et le test réel ne mesurent pas la même chose : l'écart entre leurs courbes est à interpréter avec prudence.
- SMOTE rend le modèle sensible aux pannes (sensibilité d'environ 0,8) mais génère beaucoup de fausses alertes (précision d'environ 0,6).

**Difficultés :** le très faible nombre d'exemples de certaines pannes (45 pour Tool Wear), le temps du GridSearch sur CPU et la perte de session sur Kaggle.
**Pistes d'amélioration :** variables dérivées (écart de température, puissance = couple × vitesse), poids de classes dans la loss plutôt que SMOTE complet, SMOTENC pour `Type`, jeu de validation séparé pour l'early stopping, comparaison avec Random Forest ou XGBoost.

## Exécution

```bash
pip install torch skorch scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

Ouvrir le notebook sur Kaggle (en ajoutant le dataset `shivamb/machine-predictive-maintenance-classification`) ou sur Google Colab, puis exécuter toutes les cellules.
