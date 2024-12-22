# Federated Learning System with GANs for Non-IID Client Data

## Overview
This project focuses on designing a robust federated learning system to handle challenges posed by non-IID (non-Independent and Identically Distributed) client data. The project incorporates Generative Adversarial Networks (GANs) to generate synthetic data for clients, enhancing data diversity and improving the accuracy and robustness of the global model in decentralized networks.

## Features
- **Federated Learning**: Designed a decentralized training setup where client models train locally on their data and aggregate into a global model.
- **Generative Adversarial Networks (GANs)**: Utilized GANs to generate synthetic data, addressing data heterogeneity and improving model performance.
- **Synthetic Data Augmentation**: Augmented client datasets with GAN-generated synthetic data to increase diversity and handle non-IID challenges.
- **Custom Training Pipeline**: Built a flexible pipeline for local training, model aggregation, and synthetic data generation.

## Technologies Used
- **Python**
- **PyTorch**
- **Google Colab** (for training and experimentation)
- **GANs** (Generative Adversarial Networks)
- **Federated Learning Framework**
- **Google Drive Integration** (for persistent storage)

## Project Structure
```
├── client_datasets
├── client_generators
├── synthetic_data
├── global_model
├── scripts
    ├── train_gan.py
    ├── generate_data.py
    ├── federated_training.py
```

## Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/ankitgiri577/Federated-Learning-with-GANs-for-Non-IID-Data.git
   cd Federated-Learning-with-GANs-for-Non-IID-Data
   ```
2. Install the required Python packages.
  
3. Mount your Google Drive (if using Colab).
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage
### 1. Train GANs for Client Data
Use the `train_gan.py` script to train a GAN for each client dataset.
```bash
python train_gan.py
```

### 2. Generate Synthetic Data
Generate synthetic data for clients using trained GANs.
```bash
python generate_data.py
```

### 3. Perform Federated Learning
Run the federated learning pipeline to train the global model.
```bash
python federated_training.py
```

## Key Results
- Improved global model accuracy by addressing non-IID client data challenges.
- Enhanced data diversity and model robustness using GAN-generated synthetic data.
- Demonstrated the effectiveness of synthetic data augmentation in federated setups.


## Future Work
- Deploying the federated learning system in a real-world distributed environment.
- Exploring alternative data augmentation techniques to complement GANs.
- Evaluating the system on larger, more complex datasets.
