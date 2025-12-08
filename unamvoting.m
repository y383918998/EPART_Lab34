function clab = unamvoting(tset, clsmx)
% perform unanimity voting for ensemble classifiers
% - takes test data (`tset`) and classifier matrix (`clsmx`) because we want a single predicted class per sample using all classifiers


% Simple unanimity voting function 
% 	tset - matrix containing test data; one row represents one sample
% 	clsmx - voting committee matrix

% - variables: `clab`=final predicted labels, `votes`=vote counts, 
%				`maxvotes`=votes needed for unanimity, `reject`=label for unclassified samples

% Output:
%	clab - classification result 
	
	% - determines all class labels (`labels`) depending on ensemble type (OVR if clsmx(:,2)==-1, else OVO) because OVR uses -1 for negative class
	% class processing
	if clsmx(1,2) == -1
		labels = unique(clsmx(:, 1));
		maxvotes = 1;
	else
		labels = unique(clsmx(:, [1 2]));
		maxvotes = rows(labels) - 1; % unanimity voting in one vs. one (OVO) scheme
	end
	reject = max(labels) + 1;

	% cast votes of classifiers
	% - computes votes of all classifiers using `voting()` and finds maximum votes per sample (`mv`) 
	%	---> because unanimity requires all classifiers to agree
	votes = voting(tset, clsmx);
	[mv clab] = max(votes, [], 2);

	% reject decision 
	% - applies reject decision for samples where votes do not meet unanimity threshold (`maxvotes`) 
	%	---> because we cannot classify uncertain samples confidently
	if clsmx(1,2) == -1
		clab(sum(votes, 2) ~= maxvotes) = reject;
	else
		clab(mv ~= maxvotes) = reject;
	end
end
