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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% part 1.1: feedforward
a1 = [ones(m, 1) X];
z2 = (a1 * Theta1');
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = (a2 * Theta2');
a3 = sigmoid(z3);

% part 1.2: cost function (with transformed y)
new_y = eye(num_labels)(y,:);
J = (1 / m) * sum(sum(-new_y .* log(a3) - (1 - new_y) .* log(1 - a3)));
J += (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

% part 2: back-propagation
D1 = 0;
D2 = 0;
for t = 1:m
  % 2.3.1: perform forward-propagation for a single example
  a_1 = [1 X(t,:)];
  z_2 = (a_1 * Theta1');
  a_2 = sigmoid(z_2);
  a_2 = [1 a_2];
  z_3 = (a_2 * Theta2');
  a_3 = sigmoid(z_3);

  % 2.3.2: calculate delta for output layer
  y_la = eye(num_labels)(y(t,:),:);
  d_3 = (a_3 - y_la);

  % 2.3.3: calculate delta for hidden layer 2
  d_2 = (d_3 * Theta2) .* (a_2 .* (1 - a_2));
  d_2 = d_2(2:end);

  % 2.3.4: accumulate gradients
  D2 += d_3' * a_2;
  D1 += d_2' * a_1;
end

% 2.3.5:
Theta2_grad_unreg = (1 / m) * D2;
Theta1_grad_unreg = (1 / m) * D1;

Theta2_grad_reg = (1 / m) * D2 + ((lambda / m) * Theta2);
Theta1_grad_reg = (1 / m) * D1 + ((lambda / m) * Theta1);

Theta2_grad = [ Theta2_grad_unreg(:,1) Theta2_grad_reg(:,2:end) ];
Theta1_grad = [ Theta1_grad_unreg(:,1) Theta1_grad_reg(:,2:end) ];









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
