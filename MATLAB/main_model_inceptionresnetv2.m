% Inception Resnet V2
close all; clear; clc;
timenow = datestr(now,'yyyymmdd_HHMMSS');
modelName = 'Inception-Resnet-V2';
outputFolder = 'Output';
mkdir(outputFolder)

%% === < data importing > ===
% === data source folder
data_folder = 'Data/TrainDataAll_HemBonSub/';
% === image data store to loading images
imds = imageDatastore(data_folder, ...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames',...
    'ReadFcn',@resizeTF); % using ReadFcn to reading specific image format

%% === < checking label and data balance > ===
labelCountTable = countEachLabel(imds);
labelCount = labelCountTable.Count;
min_labelCount = min(labelCount);

%% === < defining training and validation ratio > ===
% === setting ratio of training data
train_ratio = 0.7;
% === getting number of training data
numTrainFiles = fix(min_labelCount*train_ratio);
% === spliting training and validation data
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% === getting number of category
numClasses = numel(categories(imdsTrain.Labels));

%% === < showing some figures > ===
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,9);
figure
for i = 1:9
    subplot(3,3,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
% figureName = 'demo.png';
% saveas(gcf,fullfile(outputFolder,figureName))

%% === < transfer learning model > ===
TFmodel = inceptionresnetv2;
% analyzeNetwork(TFmodel)

%% === < checking image size > ===
% === reading images and getting size of them
img = readimage(imds,1);
imgSize = size(img);
% === getting size of transfer learning model
inputSize = TFmodel.Layers(1).InputSize;
% === checking image size for input images and the image input layer of the model
if imgSize == inputSize
    fprintf('Image size is %d x %d x %d.\n',imgSize(1),imgSize(2),imgSize(3))
else
    error('Image size does not match.')
end

%% === < defining layers > ===
% === defining model
lgraph = layerGraph(TFmodel);
% === removing the unwanted layers
lgraph = removeLayers(lgraph,'ClassificationLayer_predictions');
lgraph = removeLayers(lgraph,'predictions_softmax');
lgraph = removeLayers(lgraph,'predictions');
% === adding and connecting the needed layers
lgraph = addLayers(lgraph,fullyConnectedLayer(numClasses,'Name','fc_numClasses', ...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5));
lgraph = addLayers(lgraph,softmaxLayer('Name','softmax'));
lgraph = addLayers(lgraph,classificationLayer('Name','classOutput'));
lgraph = connectLayers(lgraph,'avg_pool','fc_numClasses');
lgraph = connectLayers(lgraph,'fc_numClasses','softmax');
lgraph = connectLayers(lgraph,'softmax','classOutput');
% analyzeNetwork(lgraph)

%% === < defining options to train a model > ===
% === setting mini-batch size for each iteration (could be any positive integer)
miniBatchSize = 32;
% === setting validation frequency to validate (could be any positive integer)
validationFrequency = floor(numTrainFiles/miniBatchSize);
% === setting validation patience for early stopping (could be any positive integer)
validationPatience = 20;
% === training options
% --- setting 'LearnRateSchedule' as piecewise to changing learning rate
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',validationPatience, ...
    'LearnRateSchedule','piecewise', ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto'); % 'ExecutionEnvironment': 'auto','cpu','gpu'

%% === < image data augmentation > ===
% === image data augmentation setting
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandScale',[0.9 1.0], ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30]);
imageSize = [299 299 3];
% === augmented image datastore
augimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

%% === < training network > ===
net = trainNetwork(augimds,lgraph,options);

%% === < training result > ===
% === classification
ModelPred = classify(net,imdsTrain);
% === ground truth
ModelLabel = imdsTrain.Labels;
% === predicted probability
ModelPredProb = predict(net,imdsTrain);
% === accuracy
Model_accuracy = sum(ModelPred == ModelLabel)/numel(ModelLabel);
fprintf('Training Accuracy: %.2f%%\n',100*Model_accuracy)
% === confusion matrix
figure
plotconfusion(ModelLabel,ModelPred)
title('Training Confusion Matrix')
figureName = sprintf('%s_%s_Training_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < validation result > ===
% === classification
YPred = classify(net,imdsValidation);
% === ground truth
YValidation = imdsValidation.Labels;
% === predicted probability
YPredProb = predict(net,imdsValidation);
% === accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Validation Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(YValidation,YPred)
title('Validation Confusion Matrix')
figureName = sprintf('%s_%s_Validation_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < testing result > ===
% === data source folder
test_data_src = 'Data/TestData_HemBonSub/';
% === image data store to loading images
imdsTest = imageDatastore(test_data_src, ...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames',...
    'ReadFcn',@resizeTF);
% === classification
TestPred = classify(net,imdsTest);
% === ground truth
TestLabel = imdsTest.Labels;
% === predicted probability
TestPredProb = predict(net,imdsTest);
% === accuracy
accuracy = sum(TestPred == TestLabel)/numel(TestLabel);
fprintf('Testing Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(TestLabel,TestPred)
title('Testing Confusion Matrix')
figureName = sprintf('%s_%s_Testing_ConfusionMatrix.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < ROC and AUC > ===
set(0,'defaultAxesFontSize',13)
category = {'epidural','healthy','intraparenchymal','intraventricular','subarachnoid','subdural'};

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(ModelLabel),ModelPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Training)')
figureName = sprintf('%s_%s_Training_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(YValidation),YPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Validation)')
figureName = sprintf('%s_%s_Validation_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(TestLabel),TestPredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (Testing)')
figureName = sprintf('%s_%s_Testing_ROC.png',modelName,timenow);
saveas(gcf,fullfile(outputFolder,figureName))

%% === < output > ===
filename = sprintf('%s_%s',modelName,timenow);
save(fullfile(outputFolder,filename))

disp('Finish!!!')

%%
function output = resizeTF(filename)

size = 299;
img = imread(filename);
output = imresize(img,[size size]);

end