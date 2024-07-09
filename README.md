
### Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Instructions](#instructions)
4. [Main Components](#files)
5. [Files](#filetree)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction <a name="introduction"></a>
This Repository provides a web app that detects a dog breed from an uploaded image. If a human is detected
instead of a dog, the most resembling dog breed is given. If neither a dog nor a human is detected, an error message
is displayed.

## Prerequisites<a name="prerequisites"></a>

To install the requirements in requirements.txt, run
```
pip install -r requirements.txt
```
The bottleneck features must be accessed in the directory bottleneck_features/ as a .npz-file.

The Face-Detector relies on a pre-saved Haarcascade-model, which must be stored in data/haarcascades/ as an .xml-file.

Furthermore, a pre-trained Resnet50 model and a pre-trained VGG19-model must be stored in saved_models/ as .hdf5-files.

## Instructions<a name="instructions"></a>
2. Start the Flask web app\
Provide the filepaths of the messages and categories datasets as the first and second argument respectively, as
well as the filepath of the database to save the cleaned data to as the third argument. Example:
```
python run app.py
```
Visit the Website at http://127.0.0.1:5000/

2. Run the ML-Pipeline
provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to
save the model to as the second argument. Example: 
```



## Main Components <a name="files"></a>
#### process_data.py
Reads in two .csv files, 'messages' and 'categories'. The files are merged on an ID, cleaned, and stored as SQL-table in the database DisasterResponse.db.

#### train_classifier.py
Reads the table from DisasterResponse.db and preprocesses the text messages for further use in the NLP-Pipeline. 
Since each message can be classified for multiple labels, the MultiOutputClassifier serves as a wrapper for the RandomForestClassifier.
ATF-IDF vectorizer is used for feature extraction of the text data and the model is trained using GridSearchCV. F1-Score, Precision, and Recall are reported on unseen test data 
and the trained model is saved as classifier.pkl. 

#### run.py
Generates the Flask-Web-App. The user can input a text message, which is then classified by the trained model. 
The predicted categories are highlighted. The App also shows some basic plots about the dataset that was used for training.

## Files<a name="filetree"></a>
```
.
|-- LICENSE
|-- README.md
|-- __pycache__
|   |-- extract_bottleneck_features.cpython-38.pyc
|   `-- utils.cpython-38.pyc
|-- app.py
|-- bottleneck_features
|   `-- DogVGG19Data.npz
|-- data
|   |-- haarcascades
|   |   `-- haarcascade_frontalface_alt.xml
|   `-- images
|-- dog_app.ipynb
|-- extract_bottleneck_features.py
|-- file_tree.md
|-- requirements.txt
|-- saved_models
|   |-- Resnet50.hdf5
|   `-- VGG19.hdf5
|-- templates
|   |-- display.html
|   `-- upload.html
`-- utils.py
```

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Licensed under the MIT license and provided by [Udacity](https://www.udacity.com). 

