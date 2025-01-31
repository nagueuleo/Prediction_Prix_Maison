# Prediction des prix des appartement en utilisant Straemlit et Flask


## Prediction des prix des appartement en utilisant Straemlit et Flask

Le modèle a été formé avec le jeu de données regression multiple et avec l'architecture « XGBRegressor ». Le modèle prédit le prix d'une maison donnée, l'interface utilisateur pour sélectionner les paramètres de la maison a également été construite avec Streamlit et l'API avec Flask.

## Exécuter en locale

Testez-le localement en exécutant le fichier `app.py`, créé avec `Streamlit`, et le fichier `api.py` avec `Flask`. N'oubliez pas d'exécuter d'abord le fichier `api.py`, de copier l'url http et de l'enregistrer dans la variable API du fichier `app.py`, puis de décommenter les lignes de code.

## Lancer l'application streamlit
```sh
streamlit run app.py
```

## Deploiement avec Flask
```sh
python3 api.py
```


## Resources
- Jeu de donnée: https://github.com/nagueuleo/Prediction_Prix_Maison
