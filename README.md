# Automatic Recognition of Lung Lesions and Anatomical Landmarks from Bronchoscopic Images   
*Deep learning pipelines for classification and segmentation using the BM-BronchoLC dataset*

---

## 📝 Project Description

Lung cancer remains one of the leading causes of death worldwide, with over 1.8 million deaths and 2.2 million new cases reported in 2020. Its poor 5-year survival rate is largely due to late diagnosis, highlighting the need for early and accurate lesion recognition. Bronchoscopy is a key diagnostic tool for visualizing the respiratory tract, but its interpretation is highly dependent on clinician expertise, leading to variability and missed diagnoses.

This project explores the use of artificial intelligence, particularly hybrid CNN-Transformer models, to enhance the automated recognition of pulmonary lesions and anatomical landmarks in bronchoscopic images. These hybrid models combine the local feature extraction capabilities of Convolutional Neural Networks (CNNs) with the global representation power of Transformers, leading to superior performance in medical image analysis tasks.

We leverage the **BM-BronchoLC dataset**, a large-scale and publicly available dataset specifically designed for bronchoscopic image analysis. It provides high-quality annotations for both lesion classification and anatomical landmark segmentation.

The project includes:
- The application and evaluation of **MedViT** for classification and **FCB-SwinV2 Transformer** for segmentation tasks.
- A detailed investigation of **data leakage** by comparing random and patient-wise data splitting strategies.
- Rigorous performance assessment using advanced metrics such as accuracy, AUC, Dice score, recall, and confusion matrices.

This repository provides complete pipelines for training, inference, and evaluation, aiming to support research and development in AI-assisted bronchoscopy.

---

## ✨ Main Features

1. **Comparative study of two data splitting strategies**: random vs. patient-wise  
2. **Classification of cancerous lesions** using the MedViT model  
3. **Classification of anatomical landmarks** using the MedViT model  
4. **Segmentation of cancerous lesions** with the FCB-SwinV2 Transformer  
5. **Segmentation of anatomical landmarks** with the FCB-SwinV2 Transformer

---

## 📂 Data Splitting

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

### 🔹 Random splitting — [`split_dataset.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Data_division/split_dataset.py)

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

### 🔹 Patient-wise splitting — [`split_patient.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Data_division/split_patient.py)

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

## 🛠️ 3. Classification with MedViT

### 🔹 Training – Anatomical Landmarks - [`train_medvit_Anatomical_landmarks_classification.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/anatomical_landmarks_Classification/train_medvit_Anatomical_landmarks_classification.py)

```bash
python train_medvit_Anatomical_landmarks_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --batch_size 8 \
  --epochs 200 \
  --learning_rate 1e-4 \
  --tensorboard_log_dir "./logs/anatomical" \
  --min_label_occurrences 20
```

---

### 🔹 Training – Cancerous Lesions - [`train_medvit_Lung_lesions_classification.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/lung_lesion_Classification/train_medvit_Lung_lesions_classification.py)

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

## 🧪 4. Evaluation of Classification Models

### 🔹 Testing – Anatomical Landmarks - [`test_medvit_Anatomical_landmarks_classification.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/anatomical_landmarks_Classification/test_medvit_Anatomical_landmarks_classification.py)

```bash
python test_medvit_Anatomical_landmarks_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --model_path "/path/to/trained_model.pth" \
  --output_results "results_anatomical.json" \
  --output_conf_matrix "confusion_anatomical.png"
```

---

### 🔹 Testing – Cancerous Lesions - [`test_medvit_Lung_lesions_classification.py`](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Classification/lung_lesion_Classification/test_medvit_Lung_lesions_classification.py)

```bash
python test_medvit_Lung_lesions_classification.py \
  --dataset_path "/path/to/dataset" \
  --label_json_path "/path/to/labels.json" \
  --model_path "/path/to/trained_model.pth" \
  --output_results "results_lesions.json" \
  --output_conf_matrix "confusion_lesions.png"
```

---

## 🛠️ 5. Training Segmentation Models with FCB-SwinV2 Transformer

Before training, generate the CSV files using - [create_csv_from_folder.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/anatomical_landmarks_Segmentation/create_csv_from_folder.py) (Anatomical Landmarks) - [create_csv_from_folder.py](https://github.com/Rolyph/Works_Bronchoscopy_Rolyph/blob/main/Segmentation/lung_lesion_Segmentation/create_csv_from_folder.py) (Cancerous Lesions):

```bash
python create_csv_from_folder.py \
  --dataset_path "/path/to/dataset_root"
```

---

### 🔹 Training – Anatomical Landmarks - [`Train_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py`]()

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

### 🔹 Training – Cancerous Lesions - [`Train_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py`]()

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

## 🧪 6. Evaluation of Segmentation Models

### 🔹 Testing – Anatomical Landmarks - [`Test_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py`]()

```bash
python Test_FCBSwinV2Transformer_Segmentation_Anatomical_landmarks.py \
  --test_path "/path/to/test" \
  --model_path "/path/to/best_model.pt" \
  --save_results "/path/to/save/results" \
  --image_size 384
```

---

### 🔹 Testing – Cancerous Lesions - [`Test_FCBSwinV2Transformer_Segmentation_LesionsCancereuses.py`]()

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

## 🙏 Acknowledgements

This repository includes code from the following sources:

- [BM-BronchoLC](https://github.com/csuet/bronchoscopy_nsd)  
- [MedViT](https://github.com/omid-nejati/medvit)  
- [FCB-SwinV2 Transformer](https://github.com/KerrFitzgerald/Polyp_FCB-SwinV2Transformer)

---

## 📬 Additional Information

**Contact:** Rolyph Erwan NTOUTOUME NGUEMA  
- [rolypherwan4@gmail.com](mailto:rolypherwan4@gmail.com)  
- [rolypherwanntoutoume.nguema@upoa.edu.sn](mailto:rolypherwanntoutoume.nguema@upoa.edu.sn)  
- [rolyph-erwan.ntoutoume-nguema.1@ens.etsmtl.ca](mailto:rolyph-erwan.ntoutoume-nguema.1@ens.etsmtl.ca)
