% Inception-v3
close all; clear; clc;
demo_output_folder = 'Demo';
mkdir(demo_output_folder)

%% === < loading trained model > ===
load('Output/Inception-v3_20210609_171432.mat')

%% === < importing image and classifying > ===
img_path = 'Data/TrainingDataImgPre5/epidural/ID_000edbf38.png';
img = imread(img_path);
inputSize_2D = inputSize(1:2);
img = imresize(img,inputSize_2D);

figure
[classfn,score] = classify(net,img);
imshow(img);
title(sprintf("%s (%.4f)", classfn, score(classfn)));
figureName = 'windowed_img.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < GradCam building > ===
map = gradCAM(net,img,classfn);

%% === < image and GradCam heatmap > ===
figure
imshow(img)
hold on
imagesc(map,'AlphaData',0.1)
colormap jet
hold off
title("Grad-CAM")
colorbar
figureName = 'GradCAM_img.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < importing dicom and translating to image > ===
info = dicominfo('Data/TrainingData/epidural/ID_000edbf38.dcm');
dcm_test = dicomread(info);
dcm_test = dcm_test * info.RescaleSlope + info.RescaleIntercept;
cm(dcm_test < -1000) = -1000;
side = length(dcm_test);
img_test = zeros([side,side,3]);
% === brain window
value_brain = int16( zeros([side,side]) );
loc_brain = find( 0 < dcm_test & dcm_test < 80 );
value_brain(loc_brain) = dcm_test(loc_brain);
% === image establishing
img_test(:,:,1) = value_brain;
img_test(:,:,2) = value_brain;
img_test(:,:,3) = value_brain;
img_test = uint8(img_test);
img_test = imresize(img_test,inputSize_2D);
figure
imshow(img_test)
figureName = 'dicom_brain_window.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < dicom image and GradCam heatmap > ===
figure
imshow(img_test)
hold on
imagesc(map,'AlphaData',0.1)
colormap jet
hold off
title("Grad-CAM")
colorbar
figureName = 'GradCAM_dicom.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))

%% === < dicom image and GradCam heatmap > ===
sort_mat = sort(map(:), 'descend');
cut_prob = 0.1;
cut_pt = fix(cut_prob*length(sort_mat));
mask = map > sort_mat(cut_pt);
figure
imshow(mask)
figureName = 'mask_top10per.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))
img_roi = uint8(mask).*img_test;
figure
imshow(img_roi)
figureName = 'ROI_dicom.png';
% saveas(gcf,fullfile(demo_output_folder,figureName))
