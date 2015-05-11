function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
n = length(theta);
h = sigmoid(theta' * X');

thetaReg = [0; theta(2:n)];
thetaReg(1) = 0;

% You need to return the following variables correctly
J = -(log(h) * y + log(1 - h) * (1 - y) - lambda / 2 * sum(thetaReg .^ 2)) / m;

grad = ((h - y') * X + lambda * thetaReg')' / m;

end
