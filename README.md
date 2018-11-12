## PL_Binding

Task: For each protein molecule, predict the top 10 ligands with the highest binding affinity.  
The model training is framed as a binary classification problem, according to whether a protein-ligand input pair binds or not.    
Raw data are coordinate features and atomic type. The pre-processing converts these to a 3D voxel for input into a 3D CNN.  
Execute the jupyter notebook in order, for training and evaluation.     

**Files**  
* dataset.py: preprocessing, dataloading, transformations
* train.ipynb: notebook to train CNN model
* predict.ipynb: notebook to run evaluation on the test data. requires a trained model.
* model (folder): pre-trained models can be found here
* data (folder): some toy data 

**Dependencies**  
* pytorch 0.4.0 +  

**Authors**  
* Muhammad Huzaifah
* Luis Vasquez

- - -
This project was initially written for CS5242 Neural Networks and Deep Learning project offered by the National University of Singapore. Original data was prepared by the course administrators. 






