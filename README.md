# Data Source

https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

# *MATLAB Version*

# Data Preprocessing

Different HU interval (for different tissue, i.e. bone, subdural, hemorrhage) of the three input channels

[main_preprocessing.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_preprocessing.m)

# Model Training and Evaluation

For training model via transfer learning (three nets have been tested, i.e. ResNet101, InceptionV3, InceptionResNetV2)

[main_model_resnet101.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_model_resnet101.m)

[main_model_InceptionV3.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_model_InceptionV3.m)

[main_model_inceptionresnetv2.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_model_inceptionresnetv2.m)

To output confusion matrix of the models

[main_testing_cm.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_testing_cm.m)

### Validataion

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/ResNet-101_20210609_222629_Validation_ROC.png" alt="ROC" height=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/ResNet-101_20210609_222629_Validation_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-v3_20210609_171432_Validation_ROC.png" alt="ROC" eight=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-v3_20210609_171432_Validation_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-Resnet-V2_20210609_203658_Validation_ROC.png" alt="ROC" height=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-Resnet-V2_20210609_203658_Validation_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

### Testing

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/ResNet-101_20210609_222629_Testing_ROC.png" alt="ROC" height=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/ResNet-101_20210609_222629_Testing_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-v3_20210609_171432_Testing_ROC.png" alt="ROC" eight=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-v3_20210609_171432_Testing_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-Resnet-V2_20210609_203658_Testing_ROC.png" alt="ROC" height=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/Inception-Resnet-V2_20210609_203658_Testing_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

# New Data Predicting

To predict new data from the trained models

[main_testing_predict.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_testing_predict.m)

# Ensemble Model

[Soft voting] the same weighting of the three models

[main_Multi_Classifier_System.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_Multi_Classifier_System.m)

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/MCS/MCS_Testing_ROC.png" alt="ROC" height=50% width=50%><img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/MCS/MCS_Testing_ConfusionMatrix.png" alt="Confusion Matrix" eight=50% width=50%>

# Picture Visualization

To visualize one dicom file

[main_GradCam.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_GradCam.m)

To visualize all dicom files

[main_GradCam_All.m](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/main_GradCam_All.m)

<img src="https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/MATLAB/Output/GradCAM/Inception-Resnet-V2/epidural/GradCam_Seg_ID_1b00cdf51.png" alt="Grad-CAM" eight=50% width=50%>

# *Python Version*

# Data Preprocessing

Different HU interval (for different tissue, i.e. bone, subdural, hemorrhage) of the three input channels

[main_dicom_window.ipynb](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/Python/main_dicom_window.ipynb)

[main_window_output.ipynb](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/Python/main_window_output.ipynb)

# Model Training and Evaluation

For training model via transfer learning, evaluation from confusion matrix

[main_model_InceptionV3.ipynb](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/Python/main_model_InceptionV3.ipynb)

[main_model_ResNet50_TestVersion.ipynb](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/Python/main_model_ResNet50_TestVersion.ipynb)

# Picture Visualization

[main_model_InceptionV3_GradCam_test.ipynb](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/blob/main/Python/main_model_InceptionV3_GradCam_test.ipynb)
