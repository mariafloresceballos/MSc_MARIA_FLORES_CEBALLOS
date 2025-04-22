# MSc_MARIA_FLORES_CEBALLOS

The project presented in this Master's dissertation focuses on the automatic detection of atrial fibrillation using electrocardiogram (ECG) signals through Deep Learning models.

**Main Objective**: To develop an innovative Deep Learning model for the automatic detection of atrial fibrillation from a database of real electrocardiogram recordings. The proposed approach is evaluated against other state-of-the-art Deep Learning techniques, aiming to demonstrate the effectiveness of the model in improving the clinical diagnosis of arrhythmia and preventing associated complications. The results highlight the importance of early detection of atrial fibrillation in improving patient outcomes.

## 🗂️ **Project Structure**

- 📁 `convolutional/` - folder containing ConvResNet implementation
    - 📜 `main.py` – This script contains the main logic for training and evaluating the ConvResNet model. It imports necessary libraries such as PyTorch for model implementation and custom modules like `ConvResNet`, `LazyDataset`, and `Scaler` for handling the dataset and model. The script includes functions like `fit()` for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📁 `src/` – 
        - 📜 `aux.py` - This module provides utility functions to aid model training and evaluation. It includes the `longtiming()` function, which formats time intervals into hours:minutes:seconds for better readability of training durations. It also defines the `EarlyStopper` class, which implements early stopping during training to prevent overfitting. The early stopping mechanism tracks performance improvements and halts training when the model's performance plateaus, saving computational resources and ensuring efficient training.
        - 📜 `convresnet.py` - 
        - 📜 `data.py` - This module handles the loading, preprocessing, and batching of the ECG dataset. It defines a `label()` function that processes rhythm annotations, transforming them into numerical values suitable for model training. The script also imports several libraries, including `wfdb` for handling ECG data formats, and utilizes PyTorch’s `DataLoader` and `Dataset` classes for efficient data handling. The dataset is organized and preprocessed to ensure that ECG signals and their corresponding labels are correctly aligned for model training. Additionally, it manages the storage and retrieval of data, ensuring the integrity and accessibility of ECG recordings.

- 📁 `resnext/` - folder containing ResNext implementation
    - 📜 `main.py`- This script contains the main logic for training and evaluating the ResNext model. It imports necessary libraries such as PyTorch for model implementation and custom modules like `ResNext`, `LazyDataset`, and `Scaler` for handling the dataset and model. The script includes functions like `fit()` for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📁 `src/` – 
        - 📜 `aux.py` - This module provides utility functions to aid model training and evaluation. It includes the `longtiming()` function, which formats time intervals into hours:minutes:seconds for better readability of training durations. It also defines the `EarlyStopper` class, which implements early stopping during training to prevent overfitting. The early stopping mechanism tracks performance improvements and halts training when the model's performance plateaus, saving computational resources and ensuring efficient training.
        - 📜 `resnext.py` -
        - 📜 `data.py` - This module handles the loading, preprocessing, and batching of the ECG dataset. It defines a `label()` function that processes rhythm annotations, transforming them into numerical values suitable for model training. The script also imports several libraries, including `wfdb` for handling ECG data formats, and utilizes PyTorch’s `DataLoader` and `Dataset` classes for efficient data handling. The dataset is organized and preprocessed to ensure that ECG signals and their corresponding labels are correctly aligned for model training. Additionally, it manages the storage and retrieval of data, ensuring the integrity and accessibility of ECG recordings.
- 📄 `.gitignore` – Specifies files and directories to ignore when uploading the project to GitHub.
- 📄 `LICENSE` – The project's license file, specifying usage and redistribution terms.
- 📄 `README.md` – A markdown file explaining the project’s purpose and structure.
- 📄 `requirements.txt` – Contains a list of Python dependencies and their versions needed to run the project.

## 📊 Dataset
- `Icentia11k`: Icentia11k is a large dataset consisting of continuous raw electrocardiogram (ECG) signals from 11,000 patients, recorded using the CardioSTAT™ wearable device. The data, collected between 2017 and 2018, is annotated by a team of technologists and organized into segments of approximately 70 minutes. With a sampling rate of 250 Hz and 16-bit resolution, Icentia11k provides high-quality ECG data for developing and testing Deep Learning models for arrhythmia detection in real-world, long-term monitoring scenarios.

## 📈 Results
- 📁 `results/`
    - 📓 `train_metrics` – 
    - 📓 `valid_metrics` – 
    - 📓 `test_metrics` – 

