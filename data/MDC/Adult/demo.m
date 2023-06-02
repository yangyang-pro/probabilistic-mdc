%demo of MDC baselines BR & CP with base classifiers SVM & decision tree
%SVM is implemented by built-in function fitcecoc in Matlab
%Decision tree is implemented by built-in function fitctree in Matlab
clear;clc;close all;fclose('all');
load('Adult.mat');

% main
numFolds = 10;
numAlgos = 4;
fildID = 1;%for standard output, the screen
HammingScore = zeros(numFolds,numAlgos);
ExactMatch = zeros(numFolds,numAlgos);
SubExactMatch = zeros(numFolds,numAlgos);
for numFold=1:numFolds
    %% (1)Base classifier: support vector machine
    X_train = data.norm(idx_folds{numFold}.train,:);
    X_test = data.norm(idx_folds{numFold}.test,:);
    y_train = target(idx_folds{numFold}.train,:);
    y_test = target(idx_folds{numFold}.test,:);
    
    %Binary Relevance (BR)
    iAlgo = 1;fprintf(fildID,['Fold ',num2str(numFold),': BR (svm)...\n']);
    y_predict = zeros(size(y_test));
    for dd=1:size(y_train,2)
        model_train = fitcecoc(X_train,y_train(:,dd),'Coding','onevsall'); 
        y_predict(:,dd) = predict(model_train,X_test);
    end
    %Hamming Score(or Class Accuracy)
    HammingScore(numFold,iAlgo) = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    SubExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
    
    %Class Powerset (CP)
    iAlgo = 2;fprintf(fildID,['Fold ',num2str(numFold),': CP (svm)...\n']);
    [C_y_mdc,~,ic_y_mdc] = unique(y_train,'rows');
    model_train = fitcecoc(X_train,ic_y_mdc,'Coding','onevsall');
    predicted_label_LP = predict(model_train,X_test);
    y_predict = C_y_mdc(predicted_label_LP,:);
    %Hamming Score(or Class Accuracy)
    HammingScore(numFold,iAlgo) = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    SubExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
    
    
    %% (2)Base classifier: classification decision tree
    if isempty(data.orig)
        X_train = data.norm(idx_folds{numFold}.train,:);
        X_test = data.norm(idx_folds{numFold}.test,:);
    else
        X_train = data.orig(idx_folds{numFold}.train,:);
        X_test = data.orig(idx_folds{numFold}.test,:);
    end
    y_train = target(idx_folds{numFold}.train,:);    
    y_test = target(idx_folds{numFold}.test,:);
    cat_attr = union(data_type.d_wo_o,data_type.b);%categorical features
    
	%Binary Relevance (BR)
    iAlgo = 3;fprintf(fildID,['Fold ',num2str(numFold),': BR (tree)...\n']);
    y_predict = zeros(size(y_test));
    for dd=1:size(y_train,2)
        model_train = fitctree(X_train,y_train(:,dd),'CategoricalPredictors',cat_attr); 
        y_predict(:,dd) = predict(model_train,X_test);
    end
    %Hamming Score(or Class Accuracy)
    HammingScore(numFold,iAlgo) = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    SubExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
    
    
    %Class Powerset (CP)
    iAlgo = 4;fprintf(fildID,['Fold ',num2str(numFold),': CP (tree)...\n']);
    [C_y_mdc,~,ic_y_mdc] = unique(y_train,'rows');
    model_train = fitctree(X_train,ic_y_mdc,'CategoricalPredictors',cat_attr);
    predicted_label_LP = predict(model_train,X_test);
    y_predict = C_y_mdc(predicted_label_LP,:);
    %Hamming Score(or Class Accuracy)
    HammingScore(numFold,iAlgo) = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    SubExactMatch(numFold,iAlgo) = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
end
%% disp
meanHS = mean(HammingScore);stdHS = std(HammingScore);
meanEM = mean(ExactMatch);stdEM = std(ExactMatch);
meanSEM = mean(SubExactMatch);stdSEM = std(SubExactMatch);
fprintf(fildID,'Final Results:\n');
for iAlgo=1:numAlgos
    temp_str = ['Algo.',num2str(iAlgo),': HS=',num2str(meanHS(iAlgo),'%4.3f'),'¡À',num2str(stdHS(iAlgo),'%4.3f'),...
        ', EM=',num2str(meanEM(iAlgo),'%4.3f'),'¡À',num2str(stdEM(iAlgo),'%4.3f'),...
        ', SEM=',num2str(meanSEM(iAlgo),'%4.3f'),'¡À',num2str(stdSEM(iAlgo),'%4.3f'),'\n'];
    fprintf(fildID,temp_str);
end