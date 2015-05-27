function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

n = length(lambda_vec);
error_train = zeros(n, 1);
error_val = zeros(n, 1);

for i = 1:n
    theta = trainLinearReg(X, y, lambda_vec(i));
    [error_train(i), grad_train] = linearRegCostFunction(X, y, theta, 0);
    [error_val(i), grad_val] = linearRegCostFunction(Xval, yval, theta, 0);
end

end
