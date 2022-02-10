delete all window
try
    delete(findall(0));
catch
    warning('Do not succesfully close windows.');
end
clear; clc;

%% === < confusion matrix > ===
tbl_truth = readcell('D:\NYMU\Precision_Psychiatry_Lab\Research_Topic\ICH_Detection\data\testing_filename_sortlist.xlsx');
% filename = 'Inception-v3_20201117_231043_testing_submission.xlsx';
filename = sprintf('voter_mix_%s_testing_submission.xlsx',timenow);
tbl_pred = readcell(['info_new\',filename]);
type_names = {'epidural','intraparenchymal','intraventricular','subarachnoid','subdural','healthy'};
truth = tbl_truth(:,2);
pred = tbl_pred(:,2);
cm = confusionmat(truth,pred,'Order',type_names);
accuracy = sprintf('Total Accuracy: %.2f%%',100*trace(cm)/length(pred));
truth_half = tbl_truth(1:300,2);
pred_half = tbl_pred(1:300,2);
cm_half = confusionmat(truth_half,pred_half,'Order',type_names);
cm_half_nor = 100*cm_half/50;
accuracy_half = sprintf('Accuracy of Half Data: %.2f%%',100*trace(cm_half)/length(pred_half));
fig = figure();
fig.Position(1) = 0.2*fig.Position(3);
fig.Position(2) = 0.2*fig.Position(4);
fig.Position(3) = 2.4*fig.Position(3);
fig.Position(4) = 1.2*fig.Position(4);
subplot(121)
c = confusionchart(cm_half,type_names);
c.title({'Confusion Matrix (half of data)';[accuracy,', ',accuracy_half]})
set(gca,'Fontsize',11)
subplot(122)
cn = confusionchart(cm_half_nor,type_names);
cn.title({'Normalized Confusion Matrix % (half of data)';[accuracy,', ',accuracy_half]})
set(gca,'Fontsize',11)
fig_name = [filename(1:end-23),'CM'];
saveas(fig,['info_new/ConfusionMatrix/',fig_name,'.png'])
