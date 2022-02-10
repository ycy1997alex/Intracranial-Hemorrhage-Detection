close all; clear; clc;
output_folder = 'Output/GradCAM/ResNet101/';
mkdir(output_folder)

%% === < loading trained model > ===
load('Output/ResNet-101_20210609_222629.mat')

%% === < loop > ===
label_info = dir('Data/TestData_HemBonSub/*');
label = {};
for idx_label = 1:length(label_info)-2
    label{idx_label,1} = label_info(idx_label+2).name;
end

for idx_label = 1:length(label)
    dcm_info = dir(['Data/TestingData/',label{idx_label,1},'/*.dcm']);
    img_info = dir(['Data/TestData_HemBonSub/',label{idx_label,1},'/*.png']);
    folder_out = [output_folder,label{idx_label,1}];
    mkdir(folder_out)
    fprintf('Label: %d\n',idx_label)
    parfor idx_file = 1:length(img_info)
        %% === < importing image and classifying > ===
        img_path = fullfile(img_info(idx_file).folder,img_info(idx_file).name);
        img_ht = imread(img_path);
        inputSize_2D = inputSize(1:2);
        img_ht = imresize(img_ht,inputSize_2D);
        [classfn,score] = classify(net,img_ht);
        
        %% === < GradCam building > ===
        map = gradCAM(net,img_ht,classfn);
        
        %% === < importing dicom and translating to image > ===
        info = dicominfo(fullfile(dcm_info(idx_file).folder,dcm_info(idx_file).name));
        dcm_test = dicomread(info);
        dcm_test = dcm_test * info.RescaleSlope + info.RescaleIntercept;
        dcm_test(dcm_test < -1000) = -1000;
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
        
        %% === < dicom image and GradCam heatmap > ===
        sort_mat = sort(map(:), 'descend');
        cut_prob = 0.1;
        cut_pt = fix(cut_prob*length(sort_mat));
        mask = map > sort_mat(cut_pt);
        img_roi = uint8(mask).*img_test;
        
        %% === < figure out > ===
        fig = figure('Visible','off');
        tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact');    
        
        nexttile
        imshow(img_test);
        title(sprintf("GT: %s", label{idx_label}));
        
        nexttile
        imshow(img_ht);
        title(sprintf("Pred: %s (%.4f)", classfn, score(classfn)));
        
        nexttile
        imshow(img_test)
        hold on
        imagesc(map,'AlphaData',0.1)
        colormap jet
        hold off
        title("Grad-CAM")
        colorbar
        
        nexttile
        imshow(img_roi)
        title("Segmentation")
                
        figureName = sprintf('GradCam_Seg_%s',img_info(idx_file).name);
        saveas(gcf,fullfile(folder_out,figureName))
    end
end
disp('Finish!!!')
