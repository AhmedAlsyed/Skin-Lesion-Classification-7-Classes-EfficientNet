Skin Lesion Classification with EfficientNet

Overview
This project develops an AI system for classifying seven types of skin lesions, including melanoma, using the HAM10000 dataset with 10,015 images. It aims to enable early detection of skin diseases with high accuracy using deep learning, emphasizing data leakage prevention and performance optimization through advanced techniques.
Dataset

Source: HAM10000 dataset from Kaggle.
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

Model: Pre-trained EfficientNet-B5 from torchvision, modified for 7-class classification.
Training:
Techniques: Dropout, Gradient Checkpointing, Mixed Precision Training.
Loss: CrossEntropyLoss with class weights.
Optimizer: AdamW with learning rate scheduling (CosineAnnealingWarmRestarts).
Sampling: WeightedRandomSampler to handle class imbalance.


Evaluation:
Metrics: Accuracy, Macro F1-Score, Classification Report, Confusion Matrix.
Calibration using Temperature Scaling to enhance probability reliability.



Results

Performance: Achieves >90% accuracy on the test set with a balanced F1-Score, excelling on rare classes (e.g., MEL).
Impact: Supports early skin cancer detection, with potential for real-world medical applications.
Challenges: Class imbalance, high resource consumption, and data leakage prevention.

Requirements

Language:
Libraries: PyTorch, Albumentations, scikit-learn, pandas, numpy, tqdm
Environment: Jupyter Notebook with GPU support (recommended: Kaggle or local with CUDA).
Data: HAM10000 dataset (download from Kaggle).

Installation

Clone the repository:git clone https://github.com/AhmedAlsyed/Skin-Lesion-Classification-7-Classes-EfficientNet.git


Install dependencies:pip install -r requirements.txt


Download the HAM10000 dataset from Kaggle.
Run the notebook:jupyter notebook skin-lesion-classification-7-classes-efficientne.ipynb



Usage

Ensure GPU availability for optimal performance.
Place dataset files in the appropriate paths (refer to the notebook for details).
Execute notebook cells sequentially for data preparation, training, and evaluation.

Recommendations

Improve performance using Ensemble Learning or larger models like EfficientNet-B7.
Conduct further analysis on rare classes to enhance results.
Explore additional calibration techniques for improved reliability.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.
License
This project is licensed under the MIT License.
Contact
For questions or feedback, reach out to ahmed.alsyed@example.com or open an issue on GitHub.
