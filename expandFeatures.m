function tsetExpanded = expandFeatures(tset)
% Adds additional features xi .* xj where i <= j
% tset - samples matrix (either train or test)
% tsetExpanded - samples matrix with new features (how many?)

% expand feature space with pairwise feature products
% - takes the input feature matrix (`tset`) from training or test data
% - creates new interaction features by multiplying each pair of features xi*xj (i ≤ j)
% - appends these interaction features to the original data
% - increases the dimensionality for richer linear model representation
% - variables: `ftcnt`=original feature count, `newftcnt`=total new feature count,
%              `tset`=input data matrix, `ftindex`=index for inserting new features

  ftcnt = columns(tset);
  newftcnt = ftcnt + ftcnt * (ftcnt - 1) / 2;
  tsetExpanded = [tset, zeros(rows(tset), newftcnt)];
  ftindex = columns(tset) + 1;
  for i = 1:columns(tset)
	for j = i:columns(tset)
	  tsetExpanded(:, ftindex) = tset(:,i) .* tset(:,j);
	  ftindex += 1;
	end
  end
end
