function [sp fp fn] = trainSelect(posc, negc, reps, htrain)
% - performs learning of the linear classifier `reps` times using positive (`posc`) and negative (`negc`) samples 
%	because repeated training can produce different results due to random initialization
% 	and selects the best classifier

% - uses the function handle `htrain` to compute separating plane coefficients for each repetition
% - selects the classifier with minimum weighted error (2*false positives + false negatives) because we want the most accurate separation

% 	posc - samples of class which should be on the positive side of separating plane
% 	negc - samples of class which should be on the negative side of separating plane
% 	reps - number of repetitions of training
% 	htrain - handle to function computing separating plane
% Output:
% sp - coefficients of the best separating plane
% fp - false positive count (i.e. number of misclassified samples of pclass)
% fn - false negative count (i.e. number of misclassified samples of nclass)

  manysp = zeros(reps, 1 + columns(posc));  %`manysp`=matrix storing separating plane coefficients for each repetition
  fPos = zeros(reps, 1);
  fNeg = zeros(reps, 1);
  
  for i=1:reps
    [manysp(i,:) fPos(i) fNeg(i)] = htrain(posc, negc);
  end
  
  [errCnt theBestIdx] = min(2 * fPos + fNeg);
  sp = manysp(theBestIdx, :);
  fp = fPos(theBestIdx);
  fn = fNeg(theBestIdx);
end
