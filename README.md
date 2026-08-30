# Cat Dog Classifier

A deep learning project that classifies images as cats or dogs using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Dataset](#dataset)

## ✨ Features

- Binary image classification (Cat vs Dog)
- Pre-trained Keras/TensorFlow model
- Support for multiple image formats
- Data preprocessing and validation pipeline
- Train/validation/test dataset splits

## 📁 Project Structure

```
CatDog Classifier/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── cat_dog_env/                      # Virtual environment
├── Data/                             # Dataset directory
│   ├── Raw/                          # Original dataset
│   │   ├── Cat/                      # Raw cat images
│   │   └── Dog/                      # Raw dog images
│   ├── train/                        # Training dataset
│   │   ├── Cat/
│   │   └── Dog/
│   ├── validation/                   # Validation dataset
│   │   ├── Cat/
│   │   └── Dog/
│   ├── test/                         # Test dataset
│   ├── validation_split.py           # Script for data splitting
│   ├── Duplicate_checker.py          # Script to detect duplicates
│   └── test_train_val_split.py       # Script to validate splits
└── src/                              # Source code
    ├── training.py                   # Model training script
    ├── test.py                       # Model testing/evaluation script
    ├── version.py                    # Version information
    └── dog_cat_model.h5              # Trained model file
```

## 📦 Requirements

The project requires Python 3.x with the following dependencies:

- **tensorflow** (2.17.0) - Deep learning framework
- **keras** (3.4.1) - Neural network API
- **numpy** (1.26.4) - Numerical computations
- **pandas** (2.2.2) - Data manipulation
- **Pillow** (10.4.0) - Image processing
- **scipy** (1.13.1) - Scientific computing

See [requirements.txt](requirements.txt) for the complete dependency list.

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd "CatDog Classifier"
```

### 2. Create and activate virtual environment
```bash
# On Windows
python -m venv cat_dog_env
cat_dog_env\Scripts\activate

# On macOS/Linux
python3 -m venv cat_dog_env
source cat_dog_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Prepare Dataset

Before training, organize your data and split it into train/validation/test sets:

```bash
# Split raw data into train/validation/test
cd Data
python validation_split.py

# Verify the splits
python test_train_val_split.py

# Check for duplicate images
python Duplicate_checker.py
```

### Train the Model

```bash
cd src
python training.py
```

This will:
- Load the training and validation datasets
- Train the CNN model
- Save the trained model as `dog_cat_model.h5`

### Test the Model

```bash
cd src
python test.py
```

This will:
- Load the trained model
- Evaluate performance on the test dataset
- Display classification metrics

## 🧠 Model Information

- **Model File**: `src/dog_cat_model.h5`
- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Task**: Binary Image Classification
- **Classes**: Cat (0), Dog (1)

## 📈 Model Performance (Last Training Run)

**Final Epoch Results (Epoch 30/30):**

| Metric       | Training | Validation |
|--------------|----------|------------|
| **Accuracy** | 85.38%   | 87.15%     |
| **Loss**     | 0.3331   | 0.3061     |

- Training time per step: ~328ms
- Total training steps: 318/318

**Observations:**
- The model shows strong generalization with validation accuracy (87.15%) exceeding training accuracy (85.38%)
- Validation loss (0.3061) is lower than training loss (0.3331), indicating good model performance
- No signs of overfitting; the model is learning well on both training and validation data

## 📊 Dataset

The dataset is organized into three splits:

- **Training Set** (`Data/train/`): Used for training the model
- **Validation Set** (`Data/validation/`): Used for hyperparameter tuning and early stopping
- **Test Set** (`Data/test/`): Used for final model evaluation

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.avif`

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests to improve this project.

## Future Enhancement

- improve dataset size and quality 
- EarlyStopping and ModelCheckpoin