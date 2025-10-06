Skin Lesion Classification with EfficientNet
Overview
This project develops an AI system for classifying 7 types of skin lesions (e.g., melanoma) using the HAM10000 dataset, containing 10,015 images. It aims to enable early detection of skin diseases with high accuracy using deep learning, focusing on preventing data leakage and optimizing performance through advanced techniques.
Dataset

Source: HAM10000 from Kaggle.
Classes: 7 categories (AKIEC, BCC, BKL, DF, MEL, NV, VASC).
Split:
Training: 70% (6,959 samples)
Validation: 15% (1,529 samples)
Test: 15% (1,527 samples)


Data Processing:
Unified image paths and label encoding.
Used GroupShuffleSplit to prevent data leakage via lesion_id.
Applied data augmentation with Albumentations to address class imbalance.



Methodology

Model: Pre-trained EfficientNet-B5 from torchvision, with the final layer modified for 7-class classification.
Training:
Techniques: Dropout, Gradient Checkpointing, Mixed Precision Training.
Loss: CrossEntropyLoss with class weights.
Optimizer: AdamW with learning rate scheduling (CosineAnnealingWarmRestarts).
Sampling: WeightedRandomSampler to handle class imbalance.


Evaluation:
Metrics: Accuracy, Macro F1-Score, Classification Report, Confusion Matrix.
Calibration using Temperature Scaling to improve probability reliability.



Results

Performance: Expected accuracy >90% on the test set with balanced F1-Score, particularly for rare classes (e.g., MEL).
Impact: Supports early skin cancer detection, with potential for real-world medical applications.
Challenges: Class imbalance, resource consumption, and preventing data leakage.

Requirements

Language: Python 3.11+
Libraries: PyTorch, Albumentations, scikit-learn, pandas, numpy, tqdm
Environment: Jupyter Notebook with GPU support (preferably Kaggle or local with CUDA).
Data: HAM10000 dataset (download from Kaggle).

Installation

Clone the repository:git clone https://github.com/username/repository-name.git


Install dependencies:pip install -r requirements.txt


Download the HAM10000 dataset from Kaggle.
Run the notebook:jupyter notebook skin-lesion-classification-7-classes-efficientne.ipynb



Usage

Ensure GPU availability for optimal performance.
Place dataset files in the appropriate paths (see notebook for details).
Execute notebook cells sequentially for data preparation, training, and evaluation.

Recommendations

Enhance performance with Ensemble Learning or larger models like EfficientNet-B7.
Conduct further analysis on rare classes to improve results.
Explore additional calibration techniques for better reliability.

License
MIT License
Contact
For questions or suggestions, reach out via [email] or open an issue on GitHub.
