In diesem Ordner befinden sich alle verwendeten Skripts zum Thema HOG. Das *optimize_hog_detection_model.py* Skript erledigt eigentlich alles. Es müssen nur die Parameter angepasst werden.

## *data* Ordner:
Hier sind Files wie die crosswalk-regions gespeichert

## *Optimizing* Ordner:
Hier werden die Modelle gespeichert, die trainierte werden. Ich habe nur das beste HOG Modell geteilt, da diese Modelle ansonsten zu viel Speicherplatz weggenommen hätten.

Dasselbe gilt für die vorhergesagten Label.

Zudem befinden sich in diesem Ordner die Excel Tabellen, mit den Resultaten aller Modelle, die ich zur Auswertung verwendet habe.

## *visualize_hog.py* File:
Wird verwendet, um vorhergesagte Bounding Boxes mit den richtigen Bounding Boxes zu vergleichen.