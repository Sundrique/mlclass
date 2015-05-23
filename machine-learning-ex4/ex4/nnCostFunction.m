function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];

a1 = X;
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
h = sigmoid(z3);

I = eye(size(h, 2));
y = I(y, :);

E = -(log(h) .* y + log(1 - h) .* (1 - y));

sqTheta1 =Theta1(:, 2:end).^2;
sqTheta2 =Theta2(:, 2:end).^2;

J = (sum(E(:)) + (sum(sqTheta1(:)) + sum(sqTheta2(:))) * lambda / 2) / m;

delta3 = h - y;
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

thetaReg1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
thetaReg2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = (delta2' * a1 + lambda * thetaReg1) / m;
Theta2_grad = (delta3' * a2 + lambda * thetaReg2) / m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
