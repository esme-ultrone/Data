# Data

Quelques scripts pour faciliter la collecte de data pour entraîner le réseau de neurones

* `video_recorder.py` : pour enregistrer des vidéos directement sous forme de frames jpeg, avec le même format et le même fps pour tout le monde
* `frame_extractor.py` : pour extraire les frames d'une vidéo et les grouper par label 

Le principe c'est d'enregistrer une vidéo en faisant les mouvements pour diriger le drone (takeOff, goRight, goLeft, etc..), et en suite de noter les intervalles de frames correspondant aux mouvements.  
On peut ensuite utiliser la fonction `group_frames` dans `frame_extractor` pour grouper automatiquement les frames en fonction de leur label (voir le script directement pour le fonctionnement) 
