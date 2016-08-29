function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

del1 = 0; del2 = 0; del3 = 0;
for i=1:m
    h = sigmoid(theta'*X(i,:)');
    del1 = del1 + (-y(i)*log(h)-(1-y(i))*log(1-h));
    del2 = sum(theta(2:n).^2);
    
    del3 = del3 + (h-y(i))*X(i,:)';
   
end
J = (del1/m) + (lambda/(2*m))*del2;
grad(1) = del3(1,1)/m;
grad(2:n) = (del3(2:n)/m) + (lambda/m)*theta(2:n);
% =============================================================

end
