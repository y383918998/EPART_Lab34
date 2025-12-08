function [ovosp errInfo] = trainOVOensemble(tset, tlab, htrain)
% Trains a set of linear classifiers (one versus one class)
% on the training set using trainSelect function
% tset - training set samples
% tlab - labels of the samples in the training set
% htrain - handle to proper function computing separating plane
% ovosp - one versus one class linear classifiers matrix
%   the first column contains positive class label
%   the second column contains negative class label
%   columns (3:end) contain separating plane coefficients
% errInfo - error information for individual classifiers
%   column 1 & 2 - positive and negative class labels
%   column 3 & 4 - positive & negative class misclassifications
%   column 5 & 6 - positive & negative class number of samples

  labels = unique(tlab);
  
  % nchoosek produces all possible unique pairs of labels
  % that's exactly what we need for ovo classifier
  pairs = nchoosek(labels, 2);  % This '2' produces all unique pairs of class labels, i.e., If labels = [0 1 2], then pairs = [0 1], [0 2], [1 2] --> each row is binary classifier
  ovosp = zeros(rows(pairs), 2 + 1 + columns(tset));  % here 2 is binary classifier, 1 is augmented dimension / biasness , columns(tset) is number of features in each samples
  errInfo = zeros(rows(pairs), 6);


% - variables: 
	% `labels`=unique class labels, `pairs`=all class pairs, 
	% `ovosp`=OVO classifier matrix, `errInfo`=error statistics, 
	% `htrain`=training function handle

% - generates all unique pairs of class labels (`pairs`) using nchoosek because each OVO classifier compares exactly two classes
  for i=1:rows(pairs)
	% store labels in the first two columns
    ovosp(i, 1:2) = pairs(i, :);
	
	% - iterates through each pair to select samples of the two classes from the training set (`posSamples` and `negSamples`)
    posSamples = tset(tlab == pairs(i,1), :);
    negSamples = tset(tlab == pairs(i,2), :);
	
	% - trains multiple classifiers for each pair and selects the best one (`sp`) because choosing the best reduces error for that class pair
    [sp misp misn] = trainSelect(posSamples, negSamples, 5, htrain);
	
	% what to do with errors?
	% it would be wise to add additional output argument
	% to return error coefficients
    errInfo(i, 1:2) = pairs(i, :);
    errInfo(i, 3:4) = [misp misn];
    errInfo(i, 5:6) = [rows(posSamples) rows(negSamples)];
	
    % store the separating plane coefficients (this is our classifier)
	% in ovo matrix
    ovosp(i, 3:end) = sp; 
  end
end
