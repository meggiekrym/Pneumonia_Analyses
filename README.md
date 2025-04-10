# Detection of Pneumonia with the Kaggle Dataset: Pediatric Pneumonia Chest X-ray

## Abstract
Pneumonia, a severe respiratory infection caused by bacteria, viruses, or fungi, inflames lung air sacs, affecting approximately 450 million people annually and causing 4 million deaths, with 740,180 children under 5 dying in 2019, per the World Health Organization (WHO). Symptoms like fever, cough with phlegm, extreme fatigue and breathing difficulty often start mildly but can worsen rapidly, especially in vulnerable groups such as the elderly, infants and those with weakened immune systems, though anyone can be affected.

This project investigates the use of deep learning for automatic pneumonia detection in pediatric chest X-ray images. Two models were developed and compared: a custom-designed convolutional neural network (FourLayerCNN) and a pre-trained ResNet18 adapted for grayscale images. To improve model performance, we applied data augmentation, handled class imbalance with weighted loss functions, and used an external validation set to evaluate generalization. Our results show that the custom FourLayerCNN achieved higher accuracy and better generalization than ResNet18, especially on unseen data. The findings highlight the potential of lightweight and task-specific CNN architectures for real-world medical image classification tasks.

---

## Dataset

We used two publicly available datasets from [Kaggle](https://www.kaggle.com):

  1. **Pediatric Chest X-ray Dataset** by Kermany et al.  
    - Total: 5,856 images (1,583 normal, 4,273 pneumonia)
    - Grayscale images, size 128×128  
    - Imbalanced classes

    > Note: This dataset originally comes with separate `train/` and `test/` folders.  
    > However, to ensure a clean and consistent split, we **combined the original train and test subsets** for each class (`NORMAL` and `PNEUMONIA`) into a single dataset.  
    > This allowed us to define our own training, validation, and test splits and avoid potential issues where the separation might have been pre-determined or biased. 

  2. **External Validation Set**  
    - 180 images (90 normal, 90 pneumonia)
    - Used only for final evaluation

> Due to size limitations, the full datasets are not included in this repository.  
> You can download them here:  
> [Pediatric Chest X-rays](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)  
> [External X-rays](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

---

## Model Development

We implemented and evaluated:

  - **`FourLayerCNN`**
    * 4 convolutional blocks (Conv2D → ReLU → MaxPool)
    * Dropout for regularization
    * Dense layer with 128 units
    * Final sigmoid output
    * **Trainable Parameters**: ~2.49 million

  - **`ResNet18`**
    * Pretrained on ImageNet
    * First layer adapted for grayscale input
    * Fully connected layer replaced
    * All other layers frozen (no fine-tuning)

**Techniques used:**

- Data augmentation (flip, rotate, affine transforms)
- Class imbalance handling with `pos_weight` in loss function
- Mixed precision training (AMP)
- Early stopping
- Cross-validation and external validation

---

## Results Summary

### Model Performance
| Model         | Accuracy (Test) | F1-score (Test) | Accuracy (External) | F1-score (External) |
|---------------|-----------------|-----------------|---------------------|---------------------|
| FourLayerCNN  | 94.45%          | 96.24%          | 91.11%              | 91.49%              |
| ResNet18      | 87.97%          | 92.04%          | 79.44%              | 77.02%              |

- FourLayerCNN was more stable and performed better on unseen data.
- ResNet18 showed signs of overfitting and had difficulty generalizing.

### Class-wise Performance (Internal Test Set)

#### FourLayerCNN

  | Class         | Precision     | Recall     | F1-score     |
  |---------------|---------------|------------|--------------|
  | Normal        | 0.96          | 0.79       | 0.87         |
  | Pneumonia     | 0.92          | 0.99       | 0.95         |
  | **Accuracy**  |               |            | **0.93**     |
  | Macro Avg     | 0.94          | 0.89       | 0.91         |
  | Weighted Avg  | 0.93          | 0.93       | 0.93         |

#### ResNet18

  | Class         | Precision     | Recall     | F1-score     |
  |---------------|---------------|------------|--------------|
  | Normal        | 0.88          | 0.67       | 0.76         |
  | Pneumonia     | 0.88          | 0.96       | 0.92         |
  | **Accuracy**  |               |            | **0.88**     |
  | Macro Avg     | 0.88          | 0.82       | 0.84         |
  | Weighted Avg  | 0.88          | 0.88       | 0.88         | 

---

## Installation and Usage

  1. Clone the repo:
    ```bash
    git clone https://github.com/your-repo-link.git
    cd Pneumonia_Analyses
    ```
  2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
  3. Run the Jupyter notebook:
    ```bash
    jupyter notebook xray_classification.ipynb
    ```
  4. Evaluate the model:
    ```bash
    jupyter notebook evaluate.py
    ```
  5. For more details, check the `src/` folder for the code and `models/` for pre-trained models.
  6. For the dataset, download it from the links provided above and place it in the `data/` folder.
  7. Make sure to adjust the paths in the code if necessary.

--- 

## Team Members & Contributions

  Meggie Krymowski 
  -    Data preprocessing, CNN and ResNet18, evaluation pipeline, GitHub structure, report writing

  Alexandra Schibli 
  -    Custom CNN,Data augmentation , evaluation pipeline, visualizations , report writing