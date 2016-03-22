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
alpha = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for count=1:1
  pred = sigmoid(X*theta);
  act=y;
  err1 = -y.*log(pred);
  err2= (1-y).*log(1-pred);
  diffErr=pred-act;
  logErr = err1-err2;
  grad(1)=diffErr'*X(:,1)*(alpha/m);
  grad(2)=diffErr'*X(:,2)*(alpha/m);
  grad(3)=diffErr'*X(:,3)*(alpha/m);
  J=sum(logErr)/m;
  theta=theta-grad;

  % Save the cost J in every iteration
  J_history(count,1)=count;
  J_history(count,2)=J;
% end
save itCanBeDone2.txt J_history -ascii;

% =============================================================

end
