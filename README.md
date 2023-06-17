#  MC-Dropout SSD300
The contents of this repository are the topic of my bachelor's thesis. You can find it [at this link](https://fse.studenttheses.ub.rug.nl/28357/).

This repository relies heavily on code from the the [Perception for Autonomous Systems (PAZ)](https://github.com/oarriaga/paz) library.

## SSD300 model
The model that is implemented in this repository is an Single Shot Detector with dropout layers that are on during inference time, thus enabling uncertainty quantification via model sampling. Our evaluation on the Caltech Pedestrian Dataset has shown it to successfully filter out "bogus" predictions and lead to a notable increase in mAP. However, our evaluation is limited, and we welcome further results!

![img-61](predictions/model_1/compare/img61.jpg)
![img-44](predictions/model_1/compare/img44.jpg)
![img-112](predictions/model_1/compare/img112.jpg)

## Setup
The required environment can be installed via ``` conda install environment.yml ```. If this fails, simply install based on the error messages that pop up :)

## Model usage
The simplest way to train and test the model is by using the ```train_model.py``` and ```test_model.py``` files in the root directory. 

## Using custom datasets
The provided code only supports the Caltech Pedestrian dataset in a specific format. However, if you wish to use a different dataset, please refer to the file located in ```backend/dataset_processing/prep_dataset.py```. The function ```load_data.py``` is responsible for loading the training, testing, and validation splits. Feel free to fiddle with it to load your data in the format that is required by PAZ.
