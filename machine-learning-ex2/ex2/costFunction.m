function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


h_x=sigmoid(X*theta);
log_h_x=log(h_x);
log_1_h_x = log(1-h_x);

_y_log_h_x=-y.*log_h_x;

one_y_log_1_h_x=(1-y).*log_1_h_x;

J=sum(_y_log_h_x-one_y_log_1_h_x)/m;


xx=bsxfun(@times,(h_x-y),X);
grad_sum=sum(xx);
grad=grad_sum'/m;


% =============================================================

end
