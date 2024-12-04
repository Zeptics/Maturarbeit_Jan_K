## *convert_png_to_jpeg.py* File:
Konvertiert Bilder vom png Format ins jpeg Format.

## *copy_train_valid_test_to_folder.py* File
Dieses Skript wurde verwendet, um identische Trainings- Validierungs- und Testdatensätze wie bei den NNs und den HOG Modellen zu haben.

## *create_tf_records_using_trimap.py* File:
Dieses Skript wurde verwendet, um die Trimaps zu *.tfrecord* files umzuwandeln.

## *create_trimaps_from_labelme.py* File:
Dieses wurde verwendet, um Trimaps aus den *.json* Dateien im *Labelme* Format zu erstellen, damit anschliessen die *.tfrecord* Dateien erstellt werden konnten.

## *open_trimap.py* File:
Dieses wurde verwendet, um Trimaps anzuschauen, da diese mit normalen Bildprogrammen komplett schwarz wirken, da die Werte der Pixel sehr tief sind.

## *tfrecord_reader.py* File:
Dieses wurde verwendet, um die *.tfrecord* Dateien zu lesen, da diese teils Probleme bereiteten. Da nun aber alles funktioniert, ist dieses Skript überflüssig.