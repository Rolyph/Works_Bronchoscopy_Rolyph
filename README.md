# Clinically Oriented CNN–Transformer Architectures for Reliable Bronchoscopic Recognition of Lung Lesions and Anatomical Structures  
*Deep learning pipelines for classification and segmentation using the BM-BronchoLC dataset*

---

## 📝 Project Description

Lung cancer remains one of the leading causes of cancer-related deaths worldwide, with more than **2.2 million new cases** and **1.8 million deaths** recorded in 2020. Its poor 5-year survival rate (15–20%) is mainly due to **late diagnosis** and the high variability of manual interpretation during bronchoscopy procedures.  

**Bronchoscopy** allows direct visualization of the respiratory tract but heavily relies on clinician expertise, leading to inter-operator variability and potential diagnostic errors.  

This project introduces **clinically oriented CNN–Transformer architectures** designed to **automate the recognition of anatomical structures and cancerous lesions** in bronchoscopic images, thus improving diagnostic consistency and clinical reliability.

---

##  Models Overview

Two custom hybrid models were developed and evaluated:

### 🔹 **MedViT** — *Classification Model*  
A convolution-enhanced Vision Transformer tailored for **multi-label classification** of bronchial structures and lesions.  
- Combines convolutional feature extraction with Transformer-based global attention.  
- Pre-trained on ImageNet and fine-tuned on BM-BronchoLC.  
- Achieves robust performance even under strict patient-based validation.

### 🔹 **FCB-SwinV2 Transformer** — *Segmentation Model*  
A dual-branch network coupling a **SwinV2 Transformer encoder** with a **CNN decoder**, optimized for **semantic segmentation** of bronchoscopic images.  
- Incorporates a Feature Coupling Branch (FCB) for precise boundary reconstruction.  
- Handles low-contrast and morphologically complex structures effectively.  

---

## 📊 Dataset

The models are trained and evaluated on the **[BM-BronchoLC dataset](https://doi.org/10.6084/m9.figshare.24243670.v3)** — the first public bronchoscopic dataset offering:  
- **2,921 high-quality images** from **208 patients**, acquired under standardized conditions.  
- **11 anatomical landmarks** and **7 lesion types** annotated by expert bronchoscopists.  
- High inter-observer consistency and clinically relevant labeling.  

---

## ⚙️ Key Features

- End-to-end **training and inference pipelines** for classification and segmentation.  
- Implementation in **PyTorch** with support for GPU acceleration.  
- Evaluation under **two data partitioning strategies**:  
  - *Random image-level split* — common but prone to data leakage.  
  - *Strict patient-level split* — clinically realistic and unbiased.  
- Comprehensive set of **evaluation metrics**:  
  - Accuracy, AUC-ROC, F1-score (for classification)  
  - Dice, IoU, Recall, Surface Dice (for segmentation)


---

## ✨ Code structuring 

**1. Data Splitting**  
  1.1 Random Splitting 
  
  1.2 Patient-wise Splitting 

**2. Classification with MedViT**  
  2.1 Training of Classification Models
  2.2 Evaluation of Classification Models  

**3. Segmentation with FCB-SwinV2 Transformer**  
  3.1 Training of Segmentation Models 
  3.2 Evaluation of Segmentation Models

---

## 📂 1. Data Splitting

The BM-BronchoLC dataset used in this project is publicly available at the following link:  
👉 [https://figshare.com/articles/dataset/BM-BronchoLC/24243670/3](https://figshare.com/articles/dataset/BM-BronchoLC/24243670/3)

Before splitting the data, a preprocessing pipeline must be applied, as described in the first three steps of the **"Process data"** section from the official dataset repository:  
👉 [https://github.com/csuet/bronchoscopy_nsd](https://github.com/csuet/bronchoscopy_nsd)

After preprocessing, the data should be organized in the following structure:

```
|-- Lung_cancer
|   |-- imgs
|   |   |-- images
|   |-- masks_Lung_lesions
|   |-- masks_Anatomical_landmarks
|   |-- labels_Lung_lesions.json
|   |-- labels_Anatomical_landmarks.json
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- Non_lung_cancer
|   |-- imgs
|   |-- masks_Lung_lesions
|   |-- masks_Anatomical_landmarks
|   |-- labels_Lung_lesions.json
|   |-- labels_Anatomical_landmarks.json
|   |-- annotations.json
|   |-- objects.json
|   |-- labels.json
|-- labels_Lung_lesions_final.json
|-- labels_Anatomical_landmarks_final.json
```

Once the data is ready, it can be split using one of the two strategies:

### 🔹 1.1 Random splitting — [split_dataset.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Data_division/split_dataset.py)

```bash
python split_dataset.py \
  --label_json_path "/path/to/labels.json" \
  --path_cancer_imgs "/path/to/cancer_imgs" \
  --path_non_cancer_imgs "/path/to/non_cancer_imgs" \
  --path_cancer_masks "/path/to/cancer_masks" \
  --path_non_cancer_masks "/path/to/non_cancer_masks" \
  --path_dataset "/path/to/output_dataset" \
  --task "Anatomical_landmarks"  # or "Lung_lesions"
```

---

### 🔹 1.2 Patient-wise splitting — [split_patient.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Data_division/split_patient.py)

```bash
python split_patient.py \
  --label_json_path "/path/to/labels.json" \
  --path_cancer_imgs "/path/to/cancer_imgs" \
  --path_non_cancer_imgs "/path/to/non_cancer_imgs" \
  --path_cancer_masks "/path/to/cancer_masks" \
  --path_non_cancer_masks "/path/to/non_cancer_masks" \
  --path_dataset "/path/to/output_dataset" \
  --task "Lung_lesions"  # or "Anatomical_landmarks"
```

📌 Note: You must generate two separate datasets by switching the `--task` argument.

---

## 2. Classification with MedViT

## 🛠️ 2.1 Training of Classification Models

### 🔹 2.1.1 Training – Anatomical Landmarks - [train_medvit_Anatomical_landmarks_classification.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/anatomical_landmarks_Classification/train_medvit_Anatomical_landmarks_classification.py)

```bash
python train_medvit_Anatomical_landmarks_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --batch_size 8 \
  --epochs 200 \
  --learning_rate 1e-4 \
  --tensorboard_log_dir "./logs/anatomical" \
  --min_label_occ2rrences 20
```

---

### 🔹 2.1.2 Training – Cancerous Lesions - [train_medvit_Lung_lesions_classification.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/lung_lesion_Classification/train_medvit_Lung_lesions_classification.py)

```bash
python train_medvit_Lung_lesions_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --batch_size 8 \
  --epochs 100 \
  --learning_rate 3e-5 \
  --tensorboard_log_dir "./logs/lesions" \
  --min_label_occurrences 20
```

---

## 🧪 2.2 Evaluation of Classification Models

### 🔹 2.2.1 Testing – Anatomical Landmarks - [test_medvit_Anatomical_landmarks_classification.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/anatomical_landmarks_Classification/test_medvit_Anatomical_landmarks_classification.py)

```bash
python test_medvit_Anatomical_landmarks_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --model_path "/path/to/trained_model.pth" \
  --output_results "results_anatomical.json" \
  --output_conf_matrix "confusion_anatomical.png"
```

---

### 🔹 2.2.2 Testing – Cancerous Lesions - [test_medvit_Lung_lesions_classification.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/lung_lesion_Classification/test_medvit_Lung_lesions_classification.py)

```bash
python test_medvit_Lung_lesions_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --model_path "/path/to/trained_model.pth" \
  --output_results "results_lesions.json" \
  --output_conf_matrix "confusion_lesions.png"
```

---

## 3. Segmentation with FCB-SwinV2 Transformer

Before training, generate the CSV files using - [create_csv_from_folder.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/create_csv_from_folder.py) (Anatomical Landmarks) - [create_csv_from_folder.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/create_csv_from_folder.py) (Cancerous Lesions):

```bash
python create_csv_from_folder.py \
  --dataset_path "/path/to/dataset_root"
```

---

### 🏷️ Prerequisites for Training Segmentation Models

Before executing the main script, ensure you have the following Python scripts in your working directory, depending on the segmentation task you wish to perform:

#### For anatomical landmarks segmentation:
- [FCBSWINV2_TRANSFORMER_Segmentation_Anatomical_landmarks.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/FCBSWINV2_TRANSFORMER_Segmentation_Anatomical_landmarks.py): Script implementing the segmentation model architecture
- [LOADER_Segmentation_Anatomical_landmarks.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/LOADER_Segmentation_Anatomical_landmarks.py): Script handling data loading and preprocessing
- [SIA_METRICS_Segmentation_Anatomical_landmarks.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/SIA_METRICS_Segmentation_Anatomical_landmarks.py): Script containing performance evaluation metrics

#### For cancerous lesions segmentation:
- [FCBSWINV2_TRANSFORMER_Segmentation_LesionsCancereuses.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/FCBSWINV2_TRANSFORMER_Segmentation_LesionsCancereuses.py): Implementation of the lesion-specific model
- [LOADER_Segmentation_LesionsCancereuses.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/LOADER_Segmentation_LesionsCancereuses.py): Medical data loading pipeline
- [SIA_METRICS_Segmentation_LesionsCancereuses.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/SIA_METRICS_Segmentation_LesionsCancereuses.py): Tumor-adapted evaluation metrics

---

## 🛠️ 3.1 Training of Segmentation Models

### 🔹 3.1.1 Training–Anatomical Landmarks-[Train_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/Train_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py)

```bash
python Train_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py \
  --train_path "/path/to/train" \
  --val_path "/path/to/val" \
  --save_path "/path/to/save/results" \
  --epochs 150 \
  --batch_size 2 \
  --image_size 384
```

---

### 🔹 3.1.2 Training – Cancerous Lesions - [Train_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/Train_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py)

```bash
python Train_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py \
  --train_path "/path/to/train" \
  --val_path "/path/to/val" \
  --test_path "/path/to/test" \
  --save_path "/path/to/save/results" \
  --epochs 100 \
  --batch_size 2 \
  --image_size 384
```

---

## 🧪 3.2 Evaluation of Segmentation Models

### 🔹 3.2.1 Testing – Anatomical Landmarks - [Test_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/Test_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py)

```bash
python Test_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py \
  --test_path "/path/to/test" \
  --model_path "/path/to/best_model.pt" \
  --save_results "/path/to/save/results" \
  --image_size 384
```

---

### 🔹 3.2.2 Testing – Cancerous Lesions - [Test_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/Test_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py)

```bash
python Test_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py \
  --test_path "/path/to/test" \
  --model_path "/path/to/best_model.pt" \
  --save_results "/path/to/save/results" \
  --image_size 384
```

---
---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE.txt](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/LICENSE.txt) file for more details.

---

## 🤝 Acknowledgements

- This work was conducted in collaboration between:  
  - **Université Polytechnique de l’Ouest Africain (UPOA)**, Dakar, Senegal  
  - **École de Technologie Supérieure (ÉTS)**, Montréal, Canada  

- Supported by the **Digital Research Alliance of Canada** for computational resources.

- This repository includes code from the following sources:

  - [BM-BronchoLC](https://github.com/csuet/bronchoscopy_nsd)  
  - [MedViT](https://github.com/omid-nejati/medvit)  
  - [FCB-SwinV2 Transformer](https://github.com/KerrFitzgerald/Polyp_FCB-SwinV2Transformer)


---

## 📬 Additional Information

**Contact:** Rolyph Erwan NTOUTOUME NGUEMA  
- email: [rolypherwan4@gmail.com](mailto:rolypherwan4@gmail.com)  

