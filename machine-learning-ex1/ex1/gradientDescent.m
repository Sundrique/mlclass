function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:));
J_history = zeros(num_iters, 1);

newTheta = zeros(n, 1);

for iter = 1:num_iters
    for j = 1:n
        newTheta(j) = theta(j) - alpha / m * computeDerivative(X, y, m, j, theta);
    end
    
    theta = newTheta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

function derivative = computeDerivative(X, y, m, j, theta)

derivative = 0;

for i = 1:m
    derivative = derivative + (computeHTheta(X(i,:), theta) - y(i)) * X(i, j);
end
end

function hTheta = computeHTheta(xi, theta)
    n = length(theta);
    hTheta = 0;
    for j = 1:n
        hTheta = hTheta + xi(j) * theta(j);
    end
end
