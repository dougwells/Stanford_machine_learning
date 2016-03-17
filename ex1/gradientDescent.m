function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

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

% Predicted y values with current loops thetas
pred = X*theta;
errorSquared = (pred-act).^2;     % square each element in errorSquared array
J = sum(errorSquared)/(2*m);      %sum all elements in array & divide by 2m
% fprintf('Current Loops thetas: ');
% fprintf('%f %f \n', theta(1), theta(2));
% fprintf('Cost(J) with above thetas: ');
% fprintf('%f \n', J);

% Adjust theta
    error=pred-act;                      %pred updates each loop with new thetas
    sumErrX0 = sum(error.*X(:,1));       %sum element multip. of error times Xs first col
    sumErrX1 = sum(error.*X(:,2));       %sum element multip of error times Xs second column
    adjust0 = -sumErrX0*(alpha/m);
    adjust1 = -sumErrX1*(alpha/m);
    adjust = [adjust0; adjust1];
% New thetas to use on next iteration
    theta = theta + adjust;

% Calculate J with current loops theta    
    errorSquared=error.^2;
    J = sum(errorSquared)/(2*m);

    % ============================================================

    % Save the cost J in every iteration 
    J_history(count,1) = count;   
    J_history(count,2) = computeCost(X, y, theta);
 

end
save itCanBeDone.txt J_history -ascii;
end
