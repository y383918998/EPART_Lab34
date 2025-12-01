function [ovrsp errInfo] = trainOVRensemble(tset, tlab, htrain)
% Trains a set of linear classifiers (one versus rest)
% on the training set using trainSelect function
% tset - training set samples
% tlab - labels of the samples in the training set
% htrain - handle to proper function computing separating plane
% ovrsp - one versus rest class linear classifiers matrix
%   the first column contains positive class label
%   the second column contains -1 value
%   columns (3:end) contain separating plane coefficients
% errInfo - error information for individual classifiers
%   column 1 & 2 - [positive class label -1]
%   column 3 & 4 - positive & negative class misclassifications
%   column 5 & 6 - positive & negative class number of samples

  labels = unique(tlab);
  
  ovrsp = zeros(rows(labels), 2 + 1 + columns(tset));
  errInfo = zeros(rows(labels), 6);
  
% - variables: 
	% `labels`=unique class labels, `ovrsp`=OVR classifier matrix, 
	% `errInfo`=error statistics, `htrain`=training function handle

% - generates a classifier for each class label (`labels`) because OVR separates one class from all others
  for i=1:rows(labels)
	% store label in the first column
    ovrsp(i, 1) = labels(i);
	ovrsp(i, 2) = -1;
	
	% select samples of two digits from the training set
	% - selects positive samples (`posSamples`) belonging to the current class and negative samples (`negSamples`) from all other classes
    posSamples = tset(tlab == labels(i), :);
    negSamples = tset(tlab ~= labels(i), :);
	
	% - trains multiple classifiers per class and selects the best one (`sp`) ---> because this reduces error and improves generalization
    [sp misp misn] = trainSelect(posSamples, negSamples, 5, htrain);
	
	% what to do with errors?
	% it would be wise to add additional output argument
	% to return error coefficients
    errInfo(i, 1:2) = [labels(i) -1];
    errInfo(i, 3:4) = [misp misn];
    errInfo(i, 5:6) = [rows(posSamples) rows(negSamples)];

    % store the separating plane coefficients (this is our classifier)
	% in ovr matrix
    ovrsp(i, 3:end) = sp; 
  end
end
