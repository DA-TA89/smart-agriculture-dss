# Evaluation Report — efficientnet

- Checkpoint: `C:\Project\smart-agriculture-dss\models\efficientnet_b0_plantvillage_f1_0.97.pth`
- Total parameters: 4,056,226
- Trainable parameters: 48,678
- Model size: 15.77 MB
- Inference time: 4.941 ms/image

## Overall Metrics (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.9735 |
| Precision (macro) | 0.9640 |
| Recall (macro) | 0.9687 |
| F1-score (macro) | 0.9656 |
| ROC-AUC (macro, OvR) | 0.9998 |

## ⚠️ Small-Sample Classes

The following classes have fewer test samples than the configured warning threshold (`configs/data.yaml -> split.min_samples_per_class_warning`). Interpret their per-class precision/recall with caution — small support means these numbers can swing a lot from a handful of misclassifications.

- `Potato___healthy` (support=23)

## Per-Class Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Apple___Apple_scab | 0.9592 | 1.0000 | 0.9792 | 94 |
| Apple___Black_rot | 1.0000 | 0.9892 | 0.9946 | 93 |
| Apple___Cedar_apple_rust | 0.9535 | 1.0000 | 0.9762 | 41 |
| Apple___healthy | 1.0000 | 0.9756 | 0.9877 | 246 |
| Blueberry___healthy | 1.0000 | 1.0000 | 1.0000 | 225 |
| Cherry_(including_sour)___Powdery_mildew | 1.0000 | 0.9937 | 0.9968 | 158 |
| Cherry_(including_sour)___healthy | 0.9697 | 1.0000 | 0.9846 | 128 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.7917 | 0.7403 | 0.7651 | 77 |
| Corn_(maize)___Common_rust_ | 0.9944 | 0.9888 | 0.9916 | 179 |
| Corn_(maize)___Northern_Leaf_Blight | 0.8581 | 0.8986 | 0.8779 | 148 |
| Corn_(maize)___healthy | 1.0000 | 1.0000 | 1.0000 | 174 |
| Grape___Black_rot | 1.0000 | 0.9718 | 0.9857 | 177 |
| Grape___Esca_(Black_Measles) | 0.9764 | 1.0000 | 0.9881 | 207 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | 1.0000 | 161 |
| Grape___healthy | 1.0000 | 1.0000 | 1.0000 | 63 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 0.9976 | 0.9988 | 826 |
| Peach___Bacterial_spot | 0.9885 | 0.9942 | 0.9913 | 345 |
| Peach___healthy | 0.9455 | 0.9630 | 0.9541 | 54 |
| Pepper,_bell___Bacterial_spot | 0.9934 | 1.0000 | 0.9967 | 150 |
| Pepper,_bell___healthy | 0.9821 | 0.9910 | 0.9865 | 222 |
| Potato___Early_blight | 1.0000 | 0.9733 | 0.9865 | 150 |
| Potato___Late_blight | 0.9608 | 0.9800 | 0.9703 | 150 |
| Potato___healthy | 0.9200 | 1.0000 | 0.9583 | 23 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 56 |
| Soybean___healthy | 1.0000 | 0.9935 | 0.9967 | 764 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 275 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | 1.0000 | 166 |
| Strawberry___healthy | 1.0000 | 0.9853 | 0.9926 | 68 |
| Tomato___Bacterial_spot | 0.9461 | 0.9906 | 0.9678 | 319 |
| Tomato___Early_blight | 0.8303 | 0.9133 | 0.8698 | 150 |
| Tomato___Late_blight | 0.9640 | 0.9404 | 0.9520 | 285 |
| Tomato___Leaf_Mold | 0.9776 | 0.9161 | 0.9458 | 143 |
| Tomato___Septoria_leaf_spot | 0.9553 | 0.8835 | 0.9180 | 266 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.8464 | 0.9880 | 0.9118 | 251 |
| Tomato___Target_Spot | 0.9477 | 0.7725 | 0.8512 | 211 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9950 | 0.9813 | 0.9881 | 804 |
| Tomato___Tomato_mosaic_virus | 0.9180 | 1.0000 | 0.9573 | 56 |
| Tomato___healthy | 0.9592 | 0.9874 | 0.9731 | 238 |
