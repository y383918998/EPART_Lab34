function [sepplane mispos misneg] = perceptron(pclass, nclass)
% Computes separating plane (linear classifier) using
% perceptron method.
% pclass - 'positive' class (one row contains one sample)
% nclass - 'negative' class (one row contains one sample)
% Output:
% sepplane - row vector of separating plane coefficients
% mispos - number of misclassified samples of pclass
% misneg - number of misclassified samples of nclass

  % initial random solution
  sepplane = rand(1, columns(pclass) + 1) - 0.5;

  % training data aggregation and denormalization (of the neg class)
  tset = [ ones(rows(pclass), 1) pclass; -ones(rows(nclass), 1) -nclass];
  
  nPos = rows(pclass); % number of positive samples
  nNeg = rows(nclass); % number of negative samples

  i = 1;
  do 
	%%% YOUR CODE GOES HERE %%%
	%% You should:
	%% 1. Check which samples are misclassified (boolean column vector)
    misclassified = tset * sepplane' < 0;

	%% 2. Compute separating plane correction 
	%%		This is sum of misclassfied samples coordinate times learning rate 
    delta = sum(tset(misclassified, :), 1);
	
	%% 3. Modify solution (i.e. sepplane)
    sepplane += delta;

	%% 4. Optionally you can include additional conditions to the stop criterion
	
	%%		200 iterations can take a while and probably in most cases is unnecessary

	++i;
  until i > 200;

  %%% YOUR CODE GOES HERE %%%
  %% You should:
  %% 1. Compute the numbers of false positives and false negatives

  misclassified = tset * sepplane' < 0;
  mispos = sum(misclassified(1:nPos));
  misneg = sum(misclassified(nPos+1:end));
end
