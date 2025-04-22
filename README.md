# MSc_MARIA_FLORES_CEBALLOS

The project presented in this Master's dissertation focuses on the automatic detection of atrial fibrillation using electrocardiogram (ECG) signals through Deep Learning models.

**Main Objective**: To develop an innovative Deep Learning model for the automatic detection of atrial fibrillation from a database of real electrocardiogram recordings. The proposed approach is evaluated against other state-of-the-art Deep Learning techniques, aiming to demonstrate the effectiveness of the model in improving the clinical diagnosis of arrhythmia and preventing associated complications. The results highlight the importance of early detection of atrial fibrillation in improving patient outcomes.

## 🗂️ **Project Structure**

- 📁 `convolutional/` - folder containing ConvResNet implementation
    - 📜 `main.py` – This script contains the main logic for training and evaluating the ConvResNet model. It imports necessary libraries such as PyTorch for model implementation and custom modules like `ConvResNet`, `LazyDataset`, and `Scaler` for handling the dataset and model. The script includes functions like `fit()` for training the model, logging configurations, and early stopping mechanisms to avoid overfitting. It also handles model checkpoints and integrates with metrics like accuracy, precision, recall, and F1-score to evaluate performance. The script orchestrates the training, validation, and testing phases of the model, ensuring that results are logged and saved for later analysis.
    - 📁 `src/` – 
        - 📜 `aux.py` - Contains utility functions for model training and evaluation, such as early stopping and timing functions.
        - 📜 `convresnet.py` - Defines the architecture of the ConvResNet model, specifically tailored for ECG signal classification.
        - 📜 `data.py` - Handles dataset loading, preprocessing, and batching for the ECG data.

- 📁 `resnext/` - folder containing ResNext implementation
    - 📜 `main.py`- 
    - 📁 `src/` – 
        - 📜 `aux.py` - 
        - 📜 `convresnet.py` -
        - 📜 `data.py` - 
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

