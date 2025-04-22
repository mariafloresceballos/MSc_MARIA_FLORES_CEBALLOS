# TFM_MARIA_FLORES_CEBALLOS

Genuine Carers is a project about **Large Language Models (LLM)**, **Liquid Neural Netwroks (LNN)**, **Federated Leraning** and **Jailbreak**.

**Main objective**: Design a framework based on TAI (Trustworthy AI), LMMs (Large Multimodal Models), LNNs (Liquid Neural Networks), and digital twins to create chatbots in the healthcare context. The focus is on dependent populations that are underrepresented in the data used to train standard LLMs. The goal is to work within a framework of retrainable models based on user interactions (thanks to an approach grounded in Federated Transfer Learning (FTL) and Deep Reinforcement Learning / Liquid Neural Networks (DRL/LNN)).


## 🗂️ **Project Structure**

- 📁 `convolutional/` - Conatins materials about LNN.
    - 📁 `images/` – Contains images, diagrams and model architecture visualizations for notebooks and documentation.
    - 📁 `papers/` – Collection of research papers and references relevant to the project, primarily focused on Liquid Neural Networks (LNNs).
- 📄 `.gitignore` – Specifies files and directories to ignore when uploading the project to GitHub.
- 📄 `LICENSE` – The project's open-source license file, specifying usage and redistribution terms.
- 📄 `README.md` – A markdown file explaining the project’s purpose and structure.
- 📄 `requirements.txt` – Contains a list of Python dependencies and their versions needed to run the project.

**Scripts:**  
- 📜 `DemoDDP.py` – Verifies the functionality of `torch.distributed` by performing an `all_reduce` operation between two processes.
- 📜 `DescargaModelo.py` – Downloads a pre-trained model from Hugging Face and stores it locally.


## 📊 Dataset
- `HiTZ/Multilingual-Medical-Corpus`: Multilingual-Medical-Corpus is a 3 billion word multilingual corpus for training LLMs adapted to the medical domain. Multilingual-Medical-Corpus includes four languages, namely, English, Spanish, French, and Italian. So far, we are only using the Spanish portion.

## 📈 Results
- 📁 `GPT2/`
    - 📓 `CfC_FC_GPT.ipynb` – This notebook contains the most information, runs smoothly, and is the easiest for running tests.
    - 📓 `CfC_NCP_GPT.ipynb` – The training works, but not as "fast" as the CfC Fully Connected version.
    - 📓 `LTC_FC_GPT.ipynb` – So far, OOM (Out of Memory) error. LTC layers require a huge amount of computational cost (because they need to solve differential equations), causing crashes.
    - 📓 `LTC_NCP_GPT.ipynb` – So far, OOM (Out of Memory) error. LTC layers require a huge amount of computational cost (because they need to solve differential equations), causing crashes.

- 📁 `Mistral/`
    - 📓 `CfC_FC_Mistral.ipynb` – 
