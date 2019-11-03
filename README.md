Low-Rank Matrix Factorisation
===============================
Le but de ce projet est l'implémentation d'algorithmes de recommendation selon
l'approche de collaborative-based filtering appliqué à la recommendation de films.

Dans un système de recommendation tel que Neflix ou MovieLens, on considère un
groupe d'utilisateurs et un ensemble d'objets (films dans le cas présent). Selon les
notes déjà données par les utilisateurs, il s'agit de prédire les notes qu'ils
donneraient à des nouveaux films, de telle sorte que des recommendations soient
rendues possibles.

Les techniques de collaborative filtering avec factorisation de matrices peuvent
permettent de découvrir les "attributs cachés" qui justifient les interactions
utilisateurs-objets. La factorisation de matrice avec régularisation permet d'éviter
l'overfitting.

Dans le cadre de ce projet, nous avons implémenté les techniques de factorisation suivantes:
+ Stochastic Gradient Descent (with and w/o bias)
+ Alternated Least Squares (with and w/o bias)
+ Coordinated Descent (with and w/o bias).

La factorisation de matrice est implémentée sous forme d'une classe qui contient
les différents méthodes de fit ci-dessus au sein d'un notebook Jupyter.

**Pour expérimenter avec le code:**

+ Ouvrir le notebook
+ Exécuter les deux premières cellules
+ Une fonction spécifique a été implémentée pour la lecture des données et le stockage sous forme de matrice.
```python
ratings, movies = loadData('Data'/)
R = Df2Numpy(ratings)
```
+ Pour instancier un objet, utiliser le constructeur en choisissant les paramètres
(tous ont une valeur par défaut visible dans le constructeur de la classe). Par
exemple, le code ci-dessous précise les paramètres suivants:
++ nombre de latent factors = 5
++ learning rate = 0.01
++ régularisation pour matrice utilisateurs/matrice films = 0.1
++ jeu de données divisé en 80% training et 20% validation
```python
R = Df2Numpy(ratings)
mf = MatrixFactorization(R,
    nFactors=5,
    alpha=0.01,
    lambdaReg=0.1,
    muReg=0.1,
    trainFrac=0.8,
    valFrac=0.2,
    testFrac=0.0,
    withBias=False,
    maxIter=100)
```

+ Les différentes méthodes de fit sont:
``` python
mf.stochasticGradientDescent()
mf.als()
mf.coordinateDescent()
```
+ L'exécution renvoie les trois résultats suivants
``` python
U, V, history = mf.stochasticGradientDescent(True)
```
+ Pour visualiser les résultats, utiliser
``` python
mf.plotHistory(history)
```
