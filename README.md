# AtlasFit Detection

Prototype centré sur le module de détection biceps.

## Fonctionnalités conservées

- démarrage caméra via `getUserMedia`;
- fallback par upload photo;
- image live ou capture figée;
- détection automatique des landmarks via MediaPipe Pose Landmarker;
- scan caméra ou photo d'un bras fléchi;
- détection automatique épaule, coude, poignet;
- mesure automatique bras/avant-bras;
- score de qualité de capture;
- estimation proxy biceps court/moyen/long.
- validation du premier bras puis guidage vers le deuxième bras;
- comparaison des deux bras quand les deux scans sont complétés.

## Lancer

```bash
python3 -m http.server 8000
```

Puis ouvrir `http://localhost:8000`.

Un serveur local est recommandé parce que l'accès caméra est souvent bloqué en ouverture directe `file://`.

## Prochaine étape technique

La détection automatique utilise MediaPipe Pose Landmarker côté navigateur. La classification "biceps court/long" reste un proxy: les landmarks de pose mesurent les articulations et segments, pas les insertions ni le ventre musculaire. Pour détecter réellement un muscle court/long, il faudra ajouter segmentation du contour du bras ou un modèle entraîné sur des exemples annotés.
