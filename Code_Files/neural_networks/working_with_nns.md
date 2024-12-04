## For NN with Tensorflow:
Tutorial I followed: https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
### Protocol Buffers:
Protocol Buffers should already be set up. If they for some reason aren't, do the following:

Download protocol buffers from: https://github.com/protocolbuffers/protobuf/releases (to "Tensorflow/protoc" directory).

Compile them:
1. Go to correct directory:
``` bash
cd "Python_Code\NN_bbox\Tensorflow
```
2. Compilation command:
``` bash
.\protoc\bin\protoc --"proto_path=models\research" --proto_path="models\research\object_detection\protos" "models\research\object_detection\protos\*.proto" --python_out="models\research"
```

### You have to install COCO API:
1. Ensure Visual C++ Build Tools are Installed\
   (you can get them from here: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version)

2. Install cython:
``` bash
pip install cython
```
3. Install the following:
``` bash
cd "Python_Code\NN_bbox\Tensorflow"
```
``` bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### Object Detection API Installation:
Go to the following directory:
``` bash
cd "Python_Code\NN_bbox\Tensorflow\models\research
```
Then run this:
``` bash
cp object_detection/packages/tf2/setup.py .
```
Then this:
``` bash
python -m pip install .
```
To verify installation:
``` bash
python object_detection/builders/model_builder_tf2_test.py
```
If it says OK at the end, its installed correctly
### Potential Error:
PyYaml==5.4.1 can cause Problems combined with cython==3
To install PyYaml==5.4.1 (needed by API) run the following command (potentially in Anaconda Prompt)
``` bash
pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1
```

# Insturctions for training the different models:
## Training the Neural Net:
### Convert xmls to tfrecords:
#### Convert .xml to .csv
In the "xml_to_csv.py" file located in "Python_Code/NN_bbox" put the actual path of the xml (annotation) files to the image_path variable, then run the script.

#### Convert .csv to .record
Navigate to the directory where the "generate_tfrecord_from_csv_tf2.py" file is located
``` bash
cd "Python_Code\NN_bbox"
```
Then run the following command:
``` bash
python generate_tfrecord_from_csv_tf2.py --csv_input=_labels.csv --output_path=output_path.record --image_dir=image_path_
```
Here for train, valid, test
``` bash
python generate_tfrecord_from_csv_tf2.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=Dataset/images/train

python generate_tfrecord_from_csv_tf2.py --csv_input=data/valid_labels.csv --output_path=data/valid.record --image_dir=Dataset/images/valid

python generate_tfrecord_from_csv_tf2.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=Dataset/images/test
```

### Configurate the pipeline.config file:
In the folder "Python_Code/NN_bbox/Tensorflow/workspace/models/<your_model>/vn"

Create a folder in the directory "Python_Code/NN_bbox/Tensorflow/workspace/models" with the name of your model. In that directory create a folder for every version you build. Copy a fresh pipeline.config file for every version (from "pre_trained_models/<model_name>/pipeline.config").\

**Directory Structure:**
``` bash
models/
└─ <your_model_1>/
   └─ v1/
      ├─ pipeline.config
   └─ v2/
      ├─ pipeline.config
   └─ v.../
      ├─ pipeline.config
   └─ vn/
      ├─ pipeline.config
└── ...
```
#### Parameters to adjust in pipeline.config:
**num_classes**: to the amount of classes you want to train your model for [as int]\
**batch_size** (only the one in train_config): dependant on your vram on gpu (8 is a good value to start with) [as int]\
**fine_tune_checkpoint**: Path to ckpt-0 (located in "pre_trained_models/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0") [str]\
**fine_tune_checkpoint_type**: Should be set do "detection" [str]\
**label_map_path** (for both in train- and eval_input_reader): Path to the label map ("data/label_map.pbtxt") [str]\
**input_path** (in train_input_reader to train.record file, in eval_input_reader to valid.record file): Path to corresponding file ("data/train.record" and "data/valid.record") [str]\
**total_steps** and **num_steps**: To the number of steps needed [int]\
**warmup_steps**: To the needed number < total_steps [int]\

### Training the model:
If you want to train the model on your gpu, make sure that CUDA and cuDNN are installed (here versions 11.2 and 8.1) alongside tensorflow-gpu (using pip).

If the file isn't there already, copy "model_main_tf2.py" from "Python_Code/NN_bbox/Tensorflow/models/research/object_detection/model_main_tf2.py" to the "workspace" directory.\
Then using cmd in your venv, navigate to the workspace directory.
``` bash
cd "Python_Code\NN_bbox\Tensorflow\workspace"
```
Then run the following command:
``` bash
python model_main_tf2.py --pipeline_config_path=<path to your config file> --model_dir=<path to a directory with your model> --checkpoint_every_n=<int for the number of steps per checkpoint> --num_workers=<int for the number of workers to use> --alsologtostderr
```
**--pipeline_config_path="path to config file"**: ./models/<model_name>/vn/pipeline.config\
**--model_dir="path to a directory with model"**: ./models/<model_name>/vn\
**--checkpoint_every_n="int for the number of steps per checkpoint"**: usually 100 or 1000\
**--num_workers="int for the number of workers to use"**: The amount of CPU cores the system will use to train the model\
**--alsologtostderr**: Paste as it is

**Example usage:**
``` bash
python model_main_tf2.py --pipeline_config_path=./models/efficientdet_d0_coco17_tpu-32/v1/pipeline.config --model_dir=./models/efficientdet_d0_coco17_tpu-32/v1 --checkpoint_every_n=100 --num_workers=6 --alsologtostderr
```

#### Copy checkpoint files
In order to properly evaluate the models later the checkpoint files need to be saved. Therfore head to the following directory and open the file save_all_ckpt_files.py
``` bash
cd "Python_Code\NN_bbox"
```
Then change the parameters at the top:\
**source_directory:** path leading to the version folder of the current model\
**destination_root_directory:** The path leading to the checkpoint_steps folder\
**total_steps:** The amount of steps the model will be trained for\
**last_ckpt_savings_dir:** Leave this one as it is\

Then go ahead and run the file simultaneously to the training of the model. It it is run after training it won't work.

### TensorBoard:
``` bash
--logdir=./models/<model_name>/vn
```
The path specified leads to the checkpoint files.\
Example:
``` bash
--logdir=./models/efficientdet_d0_coco17_tpu-32/v1
```

# Evaluating the Model(s)
## Setting Parameters
Open the file save_and_evaluate_models.py located in:
``` bash
cd "Python_Code\NN_bbox"
```
Then scroll all the way down and change the parameters. Most of them are pretty self-explanatory but here is a list of some:\
#### Step 1:
**already_treated_steps:** If you already ran the script for some step folders in the directory, then add them here to exclude them from doing them again (format: ['step_100', 'step_200', ...])\
**parent_directory:** Should lead to the checkpoints_steps folder\
**model_name:**: This is only for recognizing the model in the excel file later (give whatever name)\
**model_version:** In your structure your model should always be in a version folder. Put that version here (format: vn)\
**excel_file_path:** Make sure there is a prepared NN_Optimization.xlsx file there (copy the NN_Optimization_Layout file and rename it)\


**conda_environment_name:** This is to run certain commands in a python environment. Enter the name of the one you installed Tensorflow into\
**root_directory:** This should lead to your Tensorflow/models'/research/object_detection folder\
**script_name:** This is the name of the script in the root_directory that is used to save tf models (do not change)\

#### Step 2:
**input_images_directory:** The directory leading to the test images.\
**threshold_list:** A list of thresholds that should be tested (format: [0, 0.2, 0.3, ...])\

#### Step 3:
**xml_parent_dir:** In default structure the same as parent_dir\
**actual_matrix_directory:** The directory leading to the actual True/False matrices of the test images\
**specific_correct_list:** The program will look for images in which the model got n out of 16 fields correct and for images in which the model got less than n correct (format: [':4', 13, 14, ...])

## Running Script
Now in the last line the save_and_evaluate_models() function is being called. It has three input parameters (all of them booleans).
1. Parameter: saving_models (if your models aren't already saved, set to True, otherwise --> False)
2. Parameter: predicting_bboxes (same for this one)
3. Parameter: evaluating_models (same for this one)

Then finally run the script and wait for it to finish