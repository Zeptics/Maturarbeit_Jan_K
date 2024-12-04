# Erklärung des Inhalts dieses Ordners
## *Crosswalk* Ordner:
Im Ordner "Crosswalk" befindet sich ein Teil des verwendeten Datensatzes von Mike Mazurov (verfügbar am 02.12.2024 unter: https://www.kaggle.com/datasets/mikhailma/test-dataset). Zudem sind darin die Labels, die ich manuell beschriftet habe, sowohl in der Form von Bounding Boxes, als auch in der Form von Polygonen enthalten.

Die .xml Files sind die Beschriftungen in Form von Bounding Boxen (im Pascal VOC Format). Die .json Files sind die Polygon Beschriftungen im *Labelme* Format.

## *captcha_matrices* Ordner:
Hier befinden sich die Lösungen für die CAPTCHAs. In jeder Datei ist eine 4x4 Matrix für die 16 Kästchen der CAPTCHAs angegeben. *True* steht dabei für den angewählten Zustand.

## *train_val_test_split* Ordner:
### *images* Ordner:
Hier sind die 3 Ordner *train*, *val* und *test* mit den dazugehörigen Bildern gespeichert.

### *annotations* Ordner:
Hier sind die Labels der Bilder abgespeichert. Die Ordner, die auf "_bb" enden, beinhalten Bounding Boxes im Pascal VOC Format, die Ordner, die auf "_labelme" enden, beinhalten Polygone im Labelme Format und die Ordner, die auf "_trimap" enden, beinhalten die Binary Masks.
