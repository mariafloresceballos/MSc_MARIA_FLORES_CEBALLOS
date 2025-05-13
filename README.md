# MSc_MARIA_FLORES_CEBALLOS

This Master's dissertation focuses on the automatic detection of atrial fibrillation using electrocardiogram (ECG) signals through Deep Learning models.

**Main Objective**: To develop an innovative Deep Learning model for the automatic detection of atrial fibrillation from a database of real electrocardiogram recordings. The proposed approach is evaluated against other state-of-the-art Deep Learning techniques, aiming to demonstrate the effectiveness of the model in improving the clinical diagnosis of arrhythmia and preventing associated complications. The results highlight the importance of early detection of atrial fibrillation in improving patient outcomes.

## 🗂️ **Project Structure**

- 📁 `convolutional/` - folder containing ConvResNet implementation
    - 📜 `main.py` – This script contains the main logic for training and evaluating the ConvResNet model. The script includes functions like `fit()` for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📁 `src/` – 
        - 📜 `aux.py` - This module provides utility functions to aid model training and evaluation. 
        - 📜 `convresnet.py` - This module defines the architecture of the ConvResNet model, a convolutional neural network (CNN) adapted for ECG signal classification. The model consists of several ConvResNet blocks, each designed to process ECG data efficiently. The blocks are characterized by convolutional layers with kernel size 16 and subsampling mechanisms that reduce the input size by a factor of 2 in alternate residual blocks. Maxpooling is applied to the residual connections in the subsampled blocks to further reduce dimensionality. These modifications help adapt the model to handle long ECG sequences effectively, enabling it to classify different cardiac rhythms, including atrial fibrillation. The architecture's design and hyperparameters are based on the original ConvResNet paper, with some adjustments made to optimize the model's performance for the specific dataset and task at hand.
        - 📜 `data.py` - This module handles the loading, preprocessing, and batching of the ECG dataset. The dataset is organized and preprocessed to ensure that ECG signals and their corresponding labels are correctly aligned for model training. Additionally, it manages the storage and retrieval of data, ensuring the integrity and accessibility of ECG recordings.

- 📁 `resnext/` - folder containing ResNext implementation
    - 📜 `main.py`- This script contains the main logic for training and evaluating the ResNext model. The script includes functions like `fit()` for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📁 `src/` – 
        - 📜 `aux.py` - This module provides utility functions to aid model training and evaluation. 
        - 📜 `resnext.py` - This module defines the ResNeXt architecture adapted for ECG signal classification. The ResNeXt model improves upon traditional CNN architectures by using grouped convolutions, which allow it to capture more complex features while maintaining efficiency. The model is implemented with modifications tailored to process ECG data, which helps in identifying different cardiac rhythms, including atrial fibrillation. The implementation includes layers such as convolutional blocks, batch normalization, and residual connections, which enable the model to perform better on tasks involving long sequences of ECG signals.
        - 📜 `data.py` - This module handles the loading, preprocessing, and batching of the ECG dataset. The dataset is organized and preprocessed to ensure that ECG signals and their corresponding labels are correctly aligned for model training. Additionally, it manages the storage and retrieval of data, ensuring the integrity and accessibility of ECG recordings.
- 📄 `.gitignore` – Specifies files and directories to ignore when uploading the project to GitHub.
- 📄 `LICENSE` – The project's license file, specifying usage and redistribution terms.
- 📄 `README.md` – A markdown file explaining the project’s purpose and structure.
- 📄 `requirements.txt` – Contains a list of Python dependencies and their versions needed to run the project.

## 📊 Dataset
- `Icentia11k`: Icentia11k is a large dataset consisting of continuous raw electrocardiogram (ECG) signals from 11,000 patients, recorded using the CardioSTAT™ wearable device. The data, collected between 2017 and 2018, is annotated by a team of technologists and organized into segments of approximately 70 minutes. With a sampling rate of 250 Hz and 16-bit resolution, Icentia11k provides high-quality ECG data for developing and testing Deep Learning models for arrhythmia detection in real-world, long-term monitoring scenarios.

## 📈 Results
- 📁 `results/`
    - 📓 `train_metrics.csv` – This CSV file contains the training metrics for each epoch, including overall performance metrics like accuracy, precision, recall, and F1 score, as well as metrics for each individual class (e.g., atrial fibrillation, normal sinus rhythm). It helps track the model's performance during training across multiple epochs.
    - 📓 `valid_metrics.csv` – This CSV file contains the valid metrics for each epoch, including overall performance metrics like accuracy, precision, recall, and F1 score, as well as metrics for each individual class (e.g., atrial fibrillation, normal sinus rhythm). 
    - 📓 `test_metrics.csv` – This CSV file contains the test metrics for each epoch, including overall performance metrics like accuracy, precision, recall, and F1 score, as well as metrics for each individual class (e.g., atrial fibrillation, normal sinus rhythm). 

