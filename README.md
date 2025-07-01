# MSc_MARIA_FLORES_CEBALLOS

This Master's dissertation focuses on the automatic detection of atrial fibrillation using electrocardiogram (ECG) signals through Deep Learning models.

**Main Objective**: To develop an innovative Deep Learning model for the automatic detection of atrial fibrillation from a database of real electrocardiogram recordings. The proposed approach is evaluated against other state-of-the-art Deep Learning techniques, aiming to demonstrate the effectiveness of the model in improving the clinical diagnosis of arrhythmia and preventing associated complications. The results highlight the importance of early detection of atrial fibrillation in improving patient outcomes.

## 📊 Dataset
- `Icentia11k`: Icentia11k is a large dataset consisting of continuous raw electrocardiogram (ECG) signals from 11,000 patients, recorded using the CardioSTAT™ wearable device. The data, collected between 2017 and 2018, is annotated by a team of technologists and organized into segments of approximately 70 minutes. With a sampling rate of 250 Hz and 16-bit resolution, Icentia11k provides high-quality ECG data for developing and testing Deep Learning models for arrhythmia detection in real-world, long-term monitoring scenarios.


## 🗂️ **Project Structure**

- 📁 `ConvResNet/` - folder containing ConvResNet implementation
    - 📜 `main.py` – This script contains the main logic for training and evaluating the ConvResNet model. The script includes functions for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📜 `main_with_confusion_matrix.py` – Same script as before including the visualization of the confusion matrix.
    - 📜 config.yaml – This configuration file specifies paths for data, model, and scaler storage, as well as parameters for data splits, training settings, and model hyperparameters. It includes values for batch size, learning rate, weight decay, and early stopping criteria, as well as details for the model architecture such as the number of blocks and initial channels.
    - 📁 `src/` – 
        - 📜 `aux.py` - This module provides utility functions to aid model training and evaluation. 
        - 📜 `convresnet.py` - This module defines the architecture of the ConvResNet model, a convolutional neural network (CNN) adapted for ECG signal classification. The model consists of several ConvResNet blocks, each designed to process ECG data efficiently. The blocks are characterized by convolutional layers with kernel size 16 and subsampling mechanisms that reduce the input size by a factor of 2 in alternate residual blocks. Maxpooling is applied to the residual connections in the subsampled blocks to further reduce dimensionality. These modifications help adapt the model to handle long ECG sequences effectively, enabling it to classify different cardiac rhythms, including atrial fibrillation. The architecture's design and hyperparameters are based on the original ConvResNet paper, with some adjustments made to optimize the model's performance for the specific dataset and task at hand.
        - 📜 `data.py` - This module handles the loading, preprocessing, and batching of the ECG dataset. The dataset is organized and preprocessed to ensure that ECG signals and their corresponding labels are correctly aligned for model training. Additionally, it manages the storage and retrieval of data, ensuring the integrity and accessibility of ECG recordings.
        - 📜 metrics.py` - This module calculates and prints various classification metrics (accuracy, precision, recall, F1 score) for multiclass models, along with plotting precision-recall curves and loss evolution over epochs for both training and validation phases.

- 📁 `ResNeXt/` - folder containing ResNext implementation. Each subfolder contains a particular modification of the initial ResNeXt model and the same organization as explained for the ConvResNet folder.
    - 📁 `ResNext/` – Contains the initial implementation of a ResNeXt architecture inspired by the original ResNeXt-50 model adapted to study time series classification problems.
    - 📁 `ResNeXt_BO/` – Includes the implementation of Bayesian Optimization as a hyperparameter tuning for the ResNeXt model studied. To study the case before and after the data augmentation technique is applied, one has to modify the data directory in the config.yaml file.
    - 📁 `ResNeXt_GAN/` – In this folder we build a LSTM-CNN GAN as a data augmentation technique to obtain fake class 2 (atrial flutter) and 3 (atrial fibrillation) ECG fake signals to balance them. 
    - 📁 `ResNeXt_XAI/` – Finally, in this folder, Integrated Gradients are applied to explore the explainability of the ResNeXt model (before the data augmentation technique is applied).

- 📄 `.gitignore` – Specifies files and directories to ignore when uploading the project to GitHub.
- 📄 `LICENSE` – The project's license file, specifying usage and redistribution terms.
- 📄 `README.md` – A markdown file explaining the project’s purpose and structure.
- 📄 `requirements.txt` – Contains a list of Python dependencies and their versions needed to run the project.

## ⚙️ How to run the work
First of all, install the packages indicated in requirements.txt:
```bash
pip(3) install -r requirements.txt
```
In order to apply the preprocessing decribed in the work (the .zip file from PhysioNet with the original Icentia11k must be downloaded at this point):
```bash
python(3) src/data.py
```
To change the number of patients studied from the original Icentia11k database, modify number_of_patients:
```bash
for patient_path in tqdm(patient_dirs[:number_of_patients]):
```
To execute the model of interest, go to main.py file in that folder:
```bash
python(3) main.py
```
To change any configuration hyperparameters on the models, access to config.yaml file on that particular folder and save the changes before executing the main.py file.
