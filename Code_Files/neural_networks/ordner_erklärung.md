# Erkärung:
In diesem Ordner befinden sich alle zum Training eines neuronalen Netzwerkes verwendeten Skripts.

Der Arbeitsprozess war immer zweiteilig. Zuerst wurden die Modelle trainiert. Eine Anleitung wie dies funktioniert, ist im File *working_with_nns.md* zu finden.

#### Wichtig:
Wenn alle checkpoint files gespeichert werden (indem das *save_all_ckpt_files.py* File während dem Training ausgeführt wird), wird viel freier Speicherplatz benötigt. Diese checkpoints können schnell über 100 GB gross werden. Wenn diese also nicht zwingend nötig sind, rate ich davon ab, sie zu speichern.

Der 2. Teil, war die evaluierung der Modelle. Dafür ist das *save_and_evaluate_models.py* zuständig. In diesem Skript müssen ganz unten die Parameter angepasst werden, bevor es ausgeführt wird.

Im 4. Schritt kann man bestimmen, welche der drei angegebenen Prozesse ausgeführt werden sollen.

Nach dem Ausführen dieses Skripts liegen die *predictions* der Modelle in der Form von Bounding Boxes in einem Ordner mit dem Namen *predictions* vor.

## *visualize_nns.py* File:
Wird verwendet, um vorhergesagte Bounding Boxes mit den richtigen Bounding Boxes zu vergleichen.