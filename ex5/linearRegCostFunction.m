function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

pred=X*theta;
act = y;
diffErr = pred - act;
errorSquared = (diffErr).^2;	% square each element in errorSquared array
thetaAllLessZero=theta(2:end,:);
penalty = sum(thetaAllLessZero.^2)*(lambda/(2*m));
J = sum(errorSquared)/(2*m) + penalty;	%sum all elements in array & divide by 2m

thetaPenalty = thetaAllLessZero.*(lambda/m);
grad0 = diffErr'*X(:,1)./m;
gradAllLessZero = diffErr'*X(:,2:end)./m + thetaPenalty';
gradAll=[grad0,gradAllLessZero];
grad=gradAll';
theta=theta-grad;
% grad = grad(:);

% =========================================================================


end
