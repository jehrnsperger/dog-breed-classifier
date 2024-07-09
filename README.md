
![Example Image](readme_img.png)

### Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Instructions](#instructions)
4. [Main Components](#files)
5. [Files](#filetree)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction <a name="introduction"></a>
This Repository provides a web app that detects a dog breed from an uploaded image using a Convolutional Neural Network (CNN). If a human is detected
instead of a dog, the most resembling dog breed is given. If neither a dog nor a human is detected, an error message
is displayed.

The CNN was trained using Transfer Learning with Bottleneck Features. The Notebook dog_app.ipynb serves as 
provides the training of the CNN (amongst others). The file utils.py and the Flask-App in app.py are built 
on insights/models from the notebook.

## Prerequisites<a name="prerequisites"></a>

To install the requirements in requirements.txt, run
```
pip install -r requirements.txt
```
IMPORTANT: The bottleneck features are too large to be stored on Github, so you
need to store them in the directory bottleneck_features/ as a .npz-file with the name DogVGG19Data.npz. You can download the 
bottleneck features [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz).

The Face-Detector relies on a pre-saved Haarcascade-model, which must be stored in data/haarcascades/ as an .xml-file.

Furthermore, a pre-trained Resnet50 model and a pre-trained VGG19-model must be stored in saved_models/ as .hdf5-files.

## Instructions<a name="instructions"></a>
Start the Flask web app with 
```
python run app.py
```
Visit the Website at http://127.0.0.1:5000/.
Here, you can upload an image and predict the dog breed through a click.


## Main Components <a name="files"></a>
#### utils.py
Contains all the necessary functions for the web app to work. Accesses multiple pre-trained models 
and bottleneck features (see description above). 

#### dog_app.ipynb
The Jupyter Notebook which served as a starting point for this project. Here, the pre-trained models
used in utils.py are actually trained and more context is provided. The final VGG19 model achieves 
an accuracy score of about 73%. A higher score of about 83% was achieved with RestNet50; however, this
model could not be deployed due to some dependency-issues with the bottleneck features in a higher Keras version
(greater than 2.0.2). Hence, the VGG19 model was used. 

#### app.py
Generates the Flask-Web-App. The user can upload an image and the Neural Network will predict the 
dog breed (if a dog or even human is detected). 

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
Licensed under the MIT license and provided by [Udacity](https://www.udacity.com). The test picture
of the dog above originates from [phys.org](https://phys.org/news/2018-10-good-dog-canine-aptitude-clues.html).

