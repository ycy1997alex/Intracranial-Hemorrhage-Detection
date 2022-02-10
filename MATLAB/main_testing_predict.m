delete all window
try
    delete(findall(0));
catch
    warning('Do not succesfully close windows.');
end
clear; clc;

%% === < data importing > ===
% === data source folder
data_folder = './TestingDataPre6_2/';
% === image data store to loading images
imdsTest = imageDatastore(data_folder,'ReadFcn',@resizeTF); % using ReadFcn to reading specific image format

%% === < loading trained model > ===
load('info_new/Inception-Resnet-V2_20201117_024515.mat')
% load('info_new/Inception-v3_20201117_012112.mat')
% load('info_new/ResNet-101_20201116_212237.mat')
% load('info_new/Inception-v3_20201117_221647.mat')
% load('info_new/ResNet-101_20201117_220115.mat')
% load('info_new/Inception-v3_20201112_194115.mat')
% load('info_new/ResNet-101_20201111_033458.mat')
% load('info_new/Inception-v3_20201117_231043.mat')
% load('info_new/ResNet-101_20201117_205526.mat')

%% === < showing testing result > ===
numTestImages = numel(imdsTest.Files);
rng('default')
rng(2020)
idx = randperm(numTestImages,16);
% === classification
TestPred = classify(net,imdsTest);
% === probability
TestPredProb = predict(net,imdsTest);
% === fitting
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    titlename = sprintf('%s (%.2f%% to be 0)',TestPred(idx(i)),100*TestPredProb(idx(i)));
    title(titlename)
end
% === writing out
for idx_test = 1:length(imdsTest.Files)
    list{idx_test,1} = imdsTest.Files{idx_test}(end-11:end-4);
    list{idx_test,2} = char(TestPred(idx_test));
end
filename = sprintf('info_new/%s_%s_testing_submission.xlsx',modelName,timenow);
writecell(list,filename)

%%
Pre6_InceptionResnetV2 = TestPredProb;
% Pre6_InceptionV3 = TestPredProb;
% Pre6_ResNet101 = TestPredProb;
% Pre4_InceptionV3 = TestPredProb;
% Pre4_ResNet101 = TestPredProb;
% Pre2_InceptionV3 = TestPredProb;
% Pre2_ResNet101 = TestPredProb;
% Pre1_InceptionV3 = TestPredProb;
% Pre1_ResNet101 = TestPredProb;

save('voter_mix.mat',...
     'Pre6_InceptionResnetV2','Pre6_InceptionV3','Pre6_ResNet101',...
     'Pre4_InceptionV3','Pre2_InceptionV3','Pre1_InceptionV3',...
     'Pre4_ResNet101','Pre2_ResNet101','Pre1_ResNet101')

disp('Finish!!!')

%%
function output = resizeTF(filename)

% size = 224;
size = 299;
img = imread(filename);
output = imresize(img,[size size]);

end