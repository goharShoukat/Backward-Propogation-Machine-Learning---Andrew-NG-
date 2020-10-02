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
         
% You need to return the following variables correctly 

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m, 1) X];

%Forward Propogation
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';
h = sigmoid(z3);
temp_var = eye(num_labels);
y = temp_var(y,:);

J = 1/m * sum(sum(- y .* log(h) - (1-y).*log(1-h)))+...
    lambda/(2*m)*sum(sum(Theta1(:,2:end).*Theta1(:,2:end)))+...
    lambda/(2*m)*sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));

%initializations for delta matrices
%backward Propogation + regularization
delta3 = h - y;
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);


Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta2_grad = (1/m) * delta3' * a2 + lambda/m * (Theta2);
Theta1_grad = (1/m) * delta2' * X + lambda/m * (Theta1);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients


grad = [Theta1_grad(:) ; Theta2_grad(:)];

end