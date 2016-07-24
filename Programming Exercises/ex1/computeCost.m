function J = computeCost(X, y, theta)
% Doug notes: ComputeCost is a function that is used in ex1.m
% COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
% Doug: line below says to make final solution = J.  J=sumOfSquares/2m
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

m = length(X);
pred = X*theta;
act = y;
errorSquared = (pred-act).^2;	% square each element in errorSquared array
J = sum(errorSquared)/(2*m);	%sum all elements in array & divide by 2m



% =========================================================================

end
