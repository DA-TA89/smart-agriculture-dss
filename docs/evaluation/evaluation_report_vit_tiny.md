# Evaluation Report — vit

- Checkpoint: `C:\Project\smart-agriculture-dss\models\vit_tiny_plantvillage_f1_0.92.pth`
- Total parameters: 5,531,750
- Trainable parameters: 7,334
- Model size: 21.16 MB
- Inference time: 5.519 ms/image

## Overall Metrics (Test Set)

| Metric | Value |
|---|---|
| Accuracy | 0.9306 |
| Precision (macro) | 0.9201 |
| Recall (macro) | 0.9276 |
| F1-score (macro) | 0.9216 |
| ROC-AUC (macro, OvR) | 0.9991 |

## ⚠️ Small-Sample Classes

The following classes have fewer test samples than the configured warning threshold (`configs/data.yaml -> split.min_samples_per_class_warning`). Interpret their per-class precision/recall with caution — small support means these numbers can swing a lot from a handful of misclassifications.

- `Potato___healthy` (support=23)

## Per-Class Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Apple___Apple_scab | 0.9756 | 0.8511 | 0.9091 | 94 |
| Apple___Black_rot | 0.9109 | 0.9892 | 0.9485 | 93 |
| Apple___Cedar_apple_rust | 0.9762 | 1.0000 | 0.9880 | 41 |
| Apple___healthy | 0.9514 | 0.9553 | 0.9533 | 246 |
| Blueberry___healthy | 0.9657 | 1.0000 | 0.9825 | 225 |
| Cherry_(including_sour)___Powdery_mildew | 0.9515 | 0.9937 | 0.9721 | 158 |
| Cherry_(including_sour)___healthy | 1.0000 | 0.9922 | 0.9961 | 128 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.7444 | 0.8701 | 0.8024 | 77 |
| Corn_(maize)___Common_rust_ | 0.9889 | 0.9944 | 0.9916 | 179 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9124 | 0.8446 | 0.8772 | 148 |
| Corn_(maize)___healthy | 0.9886 | 0.9943 | 0.9914 | 174 |
| Grape___Black_rot | 0.9298 | 0.8983 | 0.9138 | 177 |
| Grape___Esca_(Black_Measles) | 0.9167 | 0.9565 | 0.9362 | 207 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.9875 | 0.9814 | 0.9844 | 161 |
| Grape___healthy | 0.9265 | 1.0000 | 0.9618 | 63 |
| Orange___Haunglongbing_(Citrus_greening) | 0.9988 | 0.9831 | 0.9908 | 826 |
| Peach___Bacterial_spot | 0.9740 | 0.9768 | 0.9754 | 345 |
| Peach___healthy | 0.9138 | 0.9815 | 0.9464 | 54 |
| Pepper,_bell___Bacterial_spot | 0.9351 | 0.9600 | 0.9474 | 150 |
| Pepper,_bell___healthy | 0.9604 | 0.9820 | 0.9710 | 222 |
| Potato___Early_blight | 0.9605 | 0.9733 | 0.9669 | 150 |
| Potato___Late_blight | 0.8774 | 0.9067 | 0.8918 | 150 |
| Potato___healthy | 0.7500 | 0.9130 | 0.8235 | 23 |
| Raspberry___healthy | 0.9655 | 1.0000 | 0.9825 | 56 |
| Soybean___healthy | 0.9934 | 0.9869 | 0.9902 | 764 |
| Squash___Powdery_mildew | 0.9683 | 1.0000 | 0.9839 | 275 |
| Strawberry___Leaf_scorch | 0.9758 | 0.9699 | 0.9728 | 166 |
| Strawberry___healthy | 1.0000 | 0.9853 | 0.9926 | 68 |
| Tomato___Bacterial_spot | 0.8297 | 0.9467 | 0.8843 | 319 |
| Tomato___Early_blight | 0.8261 | 0.6333 | 0.7170 | 150 |
| Tomato___Late_blight | 0.9212 | 0.7789 | 0.8441 | 285 |
| Tomato___Leaf_Mold | 0.9015 | 0.8322 | 0.8655 | 143 |
| Tomato___Septoria_leaf_spot | 0.8346 | 0.8534 | 0.8439 | 266 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.6314 | 0.9283 | 0.7516 | 251 |
| Tomato___Target_Spot | 0.7621 | 0.7441 | 0.7530 | 211 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9917 | 0.8968 | 0.9419 | 804 |
| Tomato___Tomato_mosaic_virus | 0.9074 | 0.8750 | 0.8909 | 56 |
| Tomato___healthy | 0.9606 | 0.8193 | 0.8844 | 238 |
