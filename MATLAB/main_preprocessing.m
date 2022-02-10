close all; clear; clc;

%% === < training data > ===
label_info = dir('TrainingData/*');
label = {};
for idx_label = 1:length(label_info)-2
    label{idx_label,1} = label_info(idx_label+2).name;
end

for idx_label = 1:length(label)
    file_info = dir(['TrainingData/',label{idx_label,1},'/*.dcm']);
    folder_pre = ['TrainingDataPre20210603/',label{idx_label,1}];
    mkdir(folder_pre)
    fprintf('Label: %d\n',idx_label)
    for idx_file = 1:length(file_info)
        info = dicominfo(fullfile(file_info(idx_file).folder,file_info(idx_file).name));
        dcm = dicomread(info);
        dcm = dcm * info.RescaleSlope + info.RescaleIntercept;
%         figure
%         imshow(dcm,[])
        dcm(dcm < -1000) = -1000;
        side = length(dcm);
        img = zeros([side,side,3]);
        
        value_w = int16( zeros([side,side]) );
        loc_w = find( info.WindowCenter(1)-info.WindowWidth(1) < dcm & dcm < info.WindowCenter(1)+info.WindowWidth(1) );
        value_w(loc_w) = dcm(loc_w);
        
        value_brain = int16( zeros([side,side]) );
        loc_brain = find( 0 < dcm & dcm < 80 );
        value_brain(loc_brain) = dcm(loc_brain);
        
        value_bone = int16( zeros([side,side]) );
        loc_bone = find( 600 < dcm );
        value_bone(loc_bone) = dcm(loc_bone);
        
        value_subdural = int16( zeros([side,side]) );
        loc_subdural = find( -20 < dcm & dcm < 180 );
        value_subdural(loc_subdural) = dcm(loc_subdural);
        
        value_h1 = int16( zeros([side,side]) );
        value_h2 = int16( zeros([side,side]) );
        value_h3 = int16( zeros([side,side]) );
        loc_h1 = find( 75 < dcm & dcm < 100 ); % 1 hour
        loc_h2 = find( 65 < dcm & dcm < 85 ); % 3 days
        loc_h3 = find( 35 < dcm & dcm < 40 ); % 10-14 days
        value_h1(loc_h1) = dcm(loc_h1);
        value_h2(loc_h2) = dcm(loc_h2);
        value_h3(loc_h3) = dcm(loc_h3);
        
        value_h = int16( zeros([side,side]) );
        loc_h = find( 65 < dcm & dcm < 100 ); % < 3 days
        value_h(loc_h) = dcm(loc_h);
        
        value_bone2 = int16( zeros([side,side]) );
        loc_bone2 = find( -150 < dcm & dcm < 190 );
        value_bone2(loc_bone2) = dcm(loc_bone2);
        
        value_water = int16( zeros([side,side]) );
        loc_water = find( -5 < dcm & dcm < 5 );
        value_water(loc_water) = dcm(loc_water);
        
        value_tissue = int16( zeros([side,side]) );
        loc_tissue = find( 20 < dcm & dcm < 60 );
        value_tissue(loc_tissue) = dcm(loc_tissue);
        
        img(:,:,1) = value_h;
        img(:,:,2) = value_subdural;
        img(:,:,3) = value_bone;
%         img(:,:,1) = (double(value_brain)-0) ./ 80;
%         img(:,:,2) = (double(value_subdural)-(-20)) ./ 200;
%         img(:,:,3) = (double(value_bone2)-(-150)) ./ 380;
        img_new = uint8(img);
        img_new = double(img_new)./255;
        
%         figure
%         imshow(img_new(:,:,1),[])
%         figure
%         imshow(img_new(:,:,2),[])
%         figure
%         imshow(img_new(:,:,3),[])
%         figure
%         imshow(img_new)
        
        img_name = sprintf('%s.png',fullfile(folder_pre,file_info(idx_file).name(1:end-4)));
        imwrite(img_new,img_name)
    end
end
disp('Training Data Finish!!!')

%% === < testing data > ===
file_info = dir('TestingData/*.dcm');
folder_pre = 'TestingDataPre6_2/';
mkdir(folder_pre)
for idx_file = 1:length(file_info)
    info = dicominfo(fullfile(file_info(idx_file).folder,file_info(idx_file).name));
    dcm = dicomread(info);
    dcm = dcm * info.RescaleSlope + info.RescaleIntercept;
    dcm(dcm < -1000) = -1000;
    side = length(dcm);
    img = zeros([side,side,3]);
    
    value_w = int16( zeros([side,side]) );
    loc_w = find( info.WindowCenter(1)-info.WindowWidth(1) < dcm & dcm < info.WindowCenter(1)+info.WindowWidth(1) );
    value_w(loc_w) = dcm(loc_w);
    
    value_brain = int16( zeros([side,side]) );
    loc_brain = find( 0 < dcm & dcm < 80 );
    value_brain(loc_brain) = dcm(loc_brain);
    
    value_bone = int16( zeros([side,side]) );
    loc_bone = find( 600 < dcm );
    value_bone(loc_bone) = dcm(loc_bone);
    
    value_subdural = int16( zeros([side,side]) );
    loc_subdural = find( -20 < dcm & dcm < 180 );
    value_subdural(loc_subdural) = dcm(loc_subdural);
    
    value_h1 = int16( zeros([side,side]) );
    value_h2 = int16( zeros([side,side]) );
    value_h3 = int16( zeros([side,side]) );
    loc_h1 = find( 75 < dcm & dcm < 100 ); % 1 hour
    loc_h2 = find( 65 < dcm & dcm < 85 ); % 3 days
    loc_h3 = find( 35 < dcm & dcm < 40 ); % 10-14 days
    value_h1(loc_h1) = dcm(loc_h1);
    value_h2(loc_h2) = dcm(loc_h2);
    value_h3(loc_h3) = dcm(loc_h3);
    
    value_h = int16( zeros([side,side]) );
    loc_h = find( 65 < dcm & dcm < 100 ); % < 3 days
    value_h(loc_h) = dcm(loc_h);
    
    img(:,:,1) = 3* ( value_h - (value_subdural - value_brain) );
    img(:,:,2) = value_bone;
    img(:,:,3) = 0.5*(value_brain + value_w);
    img_new = uint8(img);
    img_new = double(img_new)./255;
    
%     figure
%     imshow(img_new(:,:,1),[])
%     figure
%     imshow(img_new(:,:,2),[])
%     figure
%     imshow(img_new(:,:,3),[])
%     figure
%     imshow(img_new)
    
    img_name = sprintf('%s.png',fullfile(folder_pre,file_info(idx_file).name(1:end-4)));
    imwrite(img_new,img_name)
end
disp('Testing Data Finish!!!')