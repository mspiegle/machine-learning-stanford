function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% function [model] = svmTrain(X, Y, C, kernelFunction, tol, max_passes)

% iterate through combinations of C/sigma and store error
guesses = [];
for C_cur = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for sigma_cur = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    model = svmTrain(X, y, C_cur, @(x1, x2) gaussianKernel(x1, x2, sigma_cur));
    predictions = svmPredict(model, Xval);
    pred_error = mean(double(predictions ~= yval));
    guesses = [guesses; C_cur sigma_cur pred_error];
  end
end

% find C/sigma with lowest error
[lowest, index] = min(guesses(:, 3));
C = guesses(index, 1);
sigma = guesses(index, 2);

% =========================================================================

end
