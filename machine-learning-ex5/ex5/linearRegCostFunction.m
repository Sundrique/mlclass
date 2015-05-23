function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

thetaReg = [0; theta(2:end)];

h = X * theta;

% You need to return the following variables correctly
J = (sum((h - y) .^ 2) + sum(thetaReg .^ 2) * lambda) / 2 / m;
grad = (X' * (h - y) + lambda * thetaReg) / m;

grad = grad(:);

end
