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


%  Theta1 --> 25*401
%  Theta2 --> 10*26
%%%part 1===> cost without Regularization


y_matrix = eye(num_labels)(y,:);  % cal y matrix 5000*10
	

lay1 = [ones(m, 1) X]; % --> 5000*401

lay2 = sigmoid(lay1*Theta1'); %5000*401 X 401*25 =  5000*25

lay2_result=[ones(m, 1) lay2]; % --> 5000*26
lay3 = sigmoid(lay2_result*Theta2'); % 5000*26 X 26*10 = 5000 * 10   % use lay3 to calculate cost


log_lay3 = log(lay3);  %5000*10
log_1_lay3 = log(1-lay3);  %5000*10

_y = -1*y_matrix;  %5000*10
_1_y = 1-y_matrix;  %5000*10


J1= sum(sum(_y.*log_lay3- _1_y.*log_1_lay3))/m; % 



%%%part 2===> cost with Regularization

Theta1_ = Theta1(:,2:end);  % 25*400
Theta2_ = Theta2(:,2:end);  % 10*25

J2 = (sum(sum(Theta1_.^2)) +  sum(sum(Theta2_.^2)))*lambda/(2*m);


J = J1 + J2;


%%%part 3===> backward propagation



d3 = lay3 - y_matrix;  % 5000*10
z2 = lay1*Theta1';   %5000*401 X 401*25 =  5000*25

d2 = d3*Theta2_.*sigmoidGradient(z2) ;  % (5000*10 X 10*25) .X  5000*25 = 5000*25

delta1 = d2' * lay1; % 25*5000 X 5000*401 = 25*401
delta2 = d3' * lay2_result; % 10*5000 X 5000*26 = 10*26

Theta1_grad = delta1/m;
Theta2_grad = delta2/m;



%%%part 4===> regularized Theta gradients

Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta1_reg = Theta1_reg * (lambda/m);

Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
Theta2_reg = Theta2_reg * (lambda/m);

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
