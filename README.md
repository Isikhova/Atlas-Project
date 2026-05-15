# AtlasFit Detection

Prototype centré sur le module de détection biceps.

## Fonctionnalités conservées

- démarrage caméra via `getUserMedia`;
- fallback par upload photo;
- image live ou capture figée;
- détection automatique des landmarks via MediaPipe Pose Landmarker;
- scan caméra ou photo d'un bras fléchi;
- contrainte biceps: le bras doit être levé pour autoriser l'enregistrement automatique;
- contrainte biceps: l'angle du coude doit être compris entre 90° et 91°;
- détection automatique épaule, coude, poignet;
- mesure automatique bras/avant-bras;
- score de qualité de capture;
- estimation proxy biceps court/moyen/long.
- validation automatique du premier bras puis guidage vers le deuxième bras;
- comparaison des deux bras quand les deux scans sont complétés.
- seuil de validation biceps à 94%+ pour figer automatiquement les mesures;
- second mode de scan pour estimer le valgus des deux coudes avec les bras tendus;
- classification valgus faible/neutre, modéré ou marqué.
- comparaison gauche/droite avec écart angulaire et tolérance de capture basse.
- validation automatique du relevé valgus à 94%+ pour figer les mesures sans clic utilisateur.
- contrainte valgus: les deux paumes doivent rester ouvertes face caméra pour autoriser l'enregistrement.
- délai de calibration valgus: aucune validation automatique pendant les 15 secondes après démarrage caméra.

## Lancer

```bash
python3 -m http.server 8000
```

Puis ouvrir `http://localhost:8000`.

Un serveur local est recommandé parce que l'accès caméra est souvent bloqué en ouverture directe `file://`.

## Prochaine étape technique

La détection automatique utilise MediaPipe Pose Landmarker côté navigateur. La classification "biceps court/long" et l'estimation du valgus restent des proxys: les landmarks de pose mesurent les articulations et segments en 2D, pas les insertions ni l'axe osseux exact. Pour détecter réellement un muscle court/long ou mesurer médicalement un valgus, il faudra ajouter segmentation du contour du bras, calibration caméra ou un modèle entraîné sur des exemples annotés.
