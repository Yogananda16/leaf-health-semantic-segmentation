# Leaf Health Semantic Segmentation using SegFormer

## Overview
Manual inspection of plant health is slow and inconsistent. This project applies transformer-based deep learning to perform **semantic segmentation of leaf health**, classifying each pixel as **background**, **healthy leaf**, or **dry leaf**. The solution supports automated plant health monitoring and can be extended to broader agricultural and environmental applications.

## Dataset
- ~200 real leaf images captured using a mobile phone camera  
- Pixel-level annotations created manually using **CVAT**  
- Classes:
  - 0: Background  
  - 1: Healthy Leaf  
  - 2: Dry Leaf  
- Image resolution standardized to **512 × 512**
- Dataset split:
  - Train: 80%  
  - Validation: 10%  
  - Test: 10%

## Dataset Annotation Examples

The dataset consists of real-world images paired with manually annotated segmentation masks.

<img width="1549" height="1001" alt="Screenshot 2026-01-04 183909" src="https://github.com/user-attachments/assets/7b1422b6-1ce6-4b64-8c13-2dc4e83a74ee" />

## Methodology

### Model Architecture
- **SegFormer-B0** (Transformer-based semantic segmentation model)
- MiT (Mix Transformer) encoder pretrained on ADE20K
- Lightweight MLP decoder for dense pixel prediction
- Final segmentation head modified to output **3 classes**

<img width="1512" height="1016" alt="Screenshot 2026-01-04 183944" src="https://github.com/user-attachments/assets/08734b19-7b06-4a73-b1a2-0ef506846bd4" />

### Training Strategy
1. Baseline inference using pretrained SegFormer weights  
2. Fine-tuning using **Cross-Entropy loss**
3. Fine-tuning using combined loss:
   - 0.5 × Cross-Entropy  
   - 0.5 × Dice Loss  
4. Performance comparison using quantitative and qualitative metrics  

### Data Augmentation
- Random affine transformations  
- Brightness and contrast adjustments  
- Grid distortion and elastic transforms  
- Motion blur and noise  
- Horizontal and vertical flips  
- Coarse dropout  

---

## Evaluation Metrics
- Pixel Accuracy  
- Mean Intersection over Union (IoU)  
- Confusion Matrix  
- Qualitative visual inspection of predicted masks  

---

## Qualitative Results Before Training

The pretrained model (before fine-tuning) shows poor segmentation quality and high confusion between background and leaf regions.

<img width="1482" height="1019" alt="Screenshot 2026-01-04 184010" src="https://github.com/user-attachments/assets/c127e8ab-f019-442e-81b5-87bff5636149" />

**Observations**
- Significant background confusion  
- Poor separation of healthy vs dry leaf regions  
- Noisy and inconsistent boundaries  

---

## Results After Training (Cross-Entropy Loss)

Fine-tuning with Cross-Entropy loss leads to clear improvements in leaf localization and segmentation accuracy.

<img width="1736" height="990" alt="Screenshot 2026-01-04 184030" src="https://github.com/user-attachments/assets/5b2f0b4a-a2a8-4d49-a766-66e8ebff2a9b" />

**Key Improvements**
- Accurate identification of leaf regions  
- Strong segmentation of healthy leaves  
- Improved generalization on unseen samples  

---

## Qualitative Results After Training (CE + Dice Loss)

Introducing Dice loss significantly improves performance on minority classes and enhances boundary accuracy.

**Why Dice Loss Helps**
- Better handling of class imbalance  
- Reduced false negatives for dry leaf regions  
- Sharper and more consistent segmentation boundaries  

---

## Quantitative Results
- Validation IoU ≈ **0.83**
- Dice loss improved dry-leaf segmentation performance
- Strong generalization despite limited dataset size

<img width="1477" height="977" alt="Screenshot 2026-01-04 184053" src="https://github.com/user-attachments/assets/7ddde302-a1a8-4672-89bd-30aaedf9acd7" />


---

## Technologies Used
- Python  
- PyTorch
- Tensorflow
- SegFormer  
- Albumentations  
- OpenCV  
- NumPy  
- CVAT  

---

## Key Learnings
- Transfer learning enables strong performance on small, real-world datasets  
- Loss function selection plays a critical role in segmentation quality  
- Transformer-based architectures are effective for dense prediction tasks  
- Combining qualitative and quantitative evaluation produces reliable insights  

---

## Future Work
- Expand dataset size for improved robustness  
- Extend segmentation to additional plant disease categories  
- Deploy the trained model as an inference API for real-time use  

---
