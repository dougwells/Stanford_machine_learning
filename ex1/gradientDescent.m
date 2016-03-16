function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
    
% Calculate predicted values with initial theta
pred=X*theta;

% Just renaming y vector to make descriptive
act=y;

for count = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % run gradient descent
    % theta = gradientDescent(X, y, theta, alpha, iterations);
    % m = length(X);
    % pred = X*theta;
    % act = y;
    % errorSquared = (pred-act).^2;     % square each element in errorSquared array
    % J = sum(errorSquared)/(2*m);      %sum all elements in array & divide by 2m

% Adjust theta
    error=pred-act;
    sumErrX0 = sum(error.*X(:,1));       %sum element multip. of error times Xs first col
    sumErrX1 = sum(error.*X(:,2));       %sum element multip of error times Xs second column
    adjust0 = sumErrX0*(alpha/m);
    adjust1 = sumErrX1*(alpha/m);
    fprintf('%f %f \n', adjust0, adjust1);
    pause;

% Calculate J with new values of theta    
    errorSquared=error.^2;
    J = sum(errorSquared)/(2*m);
% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));



    % ============================================================

    % Save the cost J in every iteration    
    J_history(count) = computeCost(X, y, theta);

end

end
