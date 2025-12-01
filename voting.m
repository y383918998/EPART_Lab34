function votes = voting(tset, clsmx)
% compute votes of all one-versus-one classifiers for a dataset
% - takes test data (`tset`) and classifier matrix (`clsmx`) because we want to aggregate predictions from all (OVO) classifiers

%	clsmx(:,1) contains positive class label
%	clsmx(:,2) contains negative class label
%	clsmx(:,3) is "augmented dimension" coefficient (bias of sep. hyperplane)
%	clsmx(:,4:end) are regular separating hyperplane coefficients

% - variables: `votes`=matrix of accumulated votes, `aone`=column of ones for bias, 
%	`pid`/`nid`=indices of positive/negative classes in labels


	% get column vector of all positive labels present in the first two columns of voting ensemble
	% - because votes must be counted for each possible class
	
	%without if-else:   assumes there’s no “dummy” negative label, works well in OVO
	%with if-else: For one-vs-rest (OVR) classifiers, the second column is -1, so we only consider the first column.
	if clsmx(1,2) == -1
		labels = unique(clsmx(:, 1));
	else
		labels = unique(clsmx(:,1:2)(:));
	end

	% prepare voting result (from test data)
	votes = zeros(rows(tset), rows(labels));

	% prepare "augmented dimension" coordinate - column of "1"
	% - computes linear responses of each classifier using augmented feature dimension (bias term) because the separating hyperplane requires it
	aone = ones(size(tset,1), 1);
	
	% for all individual classifiers
	for i=1:size(clsmx,1)
		% get response of one ovo classifier for all samples
		res = [aone tset] * clsmx(i,3:end)';

		% find index of positive label of this classifier
		pid = find(labels == clsmx(i,1));

		% for all samples that produced non-negative response 
		%   increase number of votes for positive class by one
		votes(res >= 0, pid) += 1;

		% find index of negative label of this classifier
		nid = find(labels == clsmx(i,2));

		% for all samples that produced negative response 
		%   increase number of votes for negative class by one
		votes(res < 0, nid) += 1;
	end
end

