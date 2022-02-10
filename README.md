# Data Source

https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

# *MATLAB Version*

# Data Preprocessing

Different HU interval (for different tissue, i.e. bone, subdural, hemorrhage) of the three input channels

main_preprocessing.m

# Model Training

For training model via transfer learning (three nets have been tested, i.e. ResNet101, InceptionV3, InceptionResNetV2)

main_model_resnet101.m

main_model_InceptionV3.m

main_model_inceptionresnetv2.m

# Model Evaluation

To output confusion matrix of the models

main_testing_cm.m

# New Data Predicting

To predict new data from the trained models

main_testing_predict.m

# Ensemble Model

[Soft voting] the same weighting of the three models

main_Multi_Classifier_System.m

### ROC

![image](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/main/MATLAB/Output/MCS_Testing_ROC.png) 

### Confusion Matrix

![image](https://github.com/ycy1997alex/Intracranial-Hemorrhage-Detection/main/MATLAB/Output/MCS_Testing_ConfusionMatrix.png) 

# Picture Visualization

To visualize one dicom file

main_GradCam.m

To visualize all dicom files

main_GradCam_All.m

# *Python Version*

# Data Preprocessing

Different HU interval (for different tissue, i.e. bone, subdural, hemorrhage) of the three input channels

main_dicom_window.ipynb

main_window_output.ipynb

# Model Training and Evaluation

For training model via transfer learning, evaluation from confusion matrix

main_model_InceptionV3.ipynb

main_model_ResNet50_TestVersion.ipynb

# Picture Visualization

main_model_InceptionV3_GradCam_test.ipynb
