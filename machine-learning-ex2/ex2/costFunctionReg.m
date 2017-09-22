function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


h_x=sigmoid(X*theta);
log_h_x=log(h_x);
log_1_h_x = log(1-h_x);

_y_log_h_x=-y.*log_h_x;

one_y_log_1_h_x=(1-y).*log_1_h_x;

J1=sum(_y_log_h_x-one_y_log_1_h_x)/m ;

theta1 = theta; % we should exclude theat0
theta1(1) = 0;
reg_part = sum(theta1.^2)*lambda/(2*m);

J = J1 + reg_part;



temp=bsxfun(@times,(h_x-y),X);
grad_sum=sum(temp);
grad1=grad_sum'/m;


grad2 = grad1 + lambda*theta/m;
grad2(1) = grad2(1) - lambda*theta(1)/m;
grad = grad2;


% =============================================================

end
