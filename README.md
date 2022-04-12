# Classifying Movie Genres
Fatima Jangda, Noelle Lim, Eeman Salman

This repository contains the work for our ECE324 project on classifying movie genres based on movie synopses.

### Data 
The data folder contains csvs to the data we have used (3 csvs from Kaggle datasets and 1 csv from human classified movie genres). 
Also in the data folder is the code that was used to extract the respective csvs from Kaggle, concatenate all inputs (movie synopsis) and outputs (multi-class label of genre) and write it to a csv to be read and split into test and training sets for our models.

Please refer to the Jupyter notebook in the data folder titled 'DataGathering.ipynb' to see how our data was extracted, cleaned up, and concatenated. 

### Models

The model folder contains the code for the three different models we chose to implement. The CNN.py contains the code for the CNN that is still a work in progress. However the KNN.py and OneVsAll.py are fully functional models with the data created in the data folder above. The preprocess.py and utils.py files are helper files for both the KNN.py and OneVsAll.py models that help with data preprocessing as well as predicting the accuracy, precision, recall, and f-measure scores for the different tags and inputs.
