close all; clear; clc;
set(0,'defaultAxesFontSize',13)

%% === < data importing > ===
load('Output/Inception-Resnet-V2_20210609_203658.mat')
PredProb_InceptionResnetV2 = TestPredProb;
load('Output/Inception-v3_20210609_171432.mat')
PredProb_InceptionV3 = TestPredProb;
load('Output/ResNet-101_20210609_222629.mat')
PredProb_Resnet101 = TestPredProb;

%% === < one-hot encoding > ===
weight_InceptionResnetV2 = 1/3;
weight_InceptionV3       = 1/3;
weight_Resnet101         = 1/3;

%% === < weighting > ===
PredProb = weight_InceptionResnetV2 * PredProb_InceptionResnetV2 + ...
           weight_InceptionV3       * PredProb_InceptionV3       + ...
           weight_Resnet101         * PredProb_Resnet101         ;

[PredProb_value,PredProb_loc] = max(PredProb,[],2);
PredProb_type_list = {};
% epidural / healthy / intraparenchymal / intraventricular / subarachnoid / subdural
for idx = 1:length(PredProb_loc)
    PredProb_loc_idx = PredProb_loc(idx);
    if PredProb_loc_idx == 1
        PredProb_type = 'epidural';
    elseif PredProb_loc_idx == 2
        PredProb_type = 'healthy';
    elseif PredProb_loc_idx == 3
        PredProb_type = 'intraparenchymal';
    elseif PredProb_loc_idx == 4
        PredProb_type = 'intraventricular';
    elseif PredProb_loc_idx == 5
        PredProb_type = 'subarachnoid';
    elseif PredProb_loc_idx == 6
        PredProb_type = 'subdural';
    else
        error('Wrong!!!');
    end
    PredProb_type_list{idx,1} = PredProb_type;
end

Pred = categorical(PredProb_type_list);

%% === < MCS testing result > ===
outputFolder = 'Output/MCS/';
mkdir(outputFolder)
% === ground truth
Label = TestLabel;
% === accuracy
accuracy = sum(Pred == Label)/numel(Label);
fprintf('Testing Accuracy: %.2f%%\n',100*accuracy)
% === confusion matrix
figure
plotconfusion(Label,Pred)
title('Testing Confusion Matrix (MCS)')
figureName = sprintf('MCS_Testing_ConfusionMatrix.png');
saveas(gcf,fullfile(outputFolder,figureName))
% === evaluation
confusionMat = confusionmat(Label,Pred);
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
meanPrecision = @(confusionMat) mean(precision(confusionMat));
meanRecall = @(confusionMat) mean(recall(confusionMat));
meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
fprintf('Precision: %.4f\n',precision(confusionMat))
fprintf('Recall: %.4f\n',recall(confusionMat))
fprintf('F1 Scores: %.4f\n',f1Scores(confusionMat))
fprintf('Mean Precision: %.4f\n',meanPrecision(confusionMat))
fprintf('Mean Recall: %.4f\n',meanRecall(confusionMat))
fprintf('Mean F1 Score: %.4f\n',meanF1(confusionMat))
% === ROC and AUC
category = {'epidural','healthy','intraparenchymal','intraventricular','subarachnoid','subdural'};
figure
for idx = 1:length(category)
    [X,Y,T,AUC] = perfcurve(cellstr(Label),PredProb(:,idx),category{idx});
    displayname = sprintf('%s - AUC: %.4f',category{idx},AUC);
    plot(X,Y,'DisplayName',displayname)
    hold on
end
legend('Location','SouthEast')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification (MCS Testing)')
figureName = sprintf('MCS_Testing_ROC.png');
saveas(gcf,fullfile(outputFolder,figureName))
