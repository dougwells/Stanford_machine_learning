function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% !!! Change to the following values when "submitting"
%     alpha = 1;
%     loopend = 1;
% Note: grad should have the same dimensions as theta
%
alpha=1;
loopEnd=1;
%
for count=1:loopEnd;
  pred = sigmoid(X*theta);
  act=y;
  err1 = -y.*log(pred);
  err2= (1-y).*log(1-pred);
  diffErr=pred-act;
  logErr = err1-err2;
  thetaZero = theta(1,:);
  thetaAllLessZero=theta(2:end,:);
  penalty = sum(thetaAllLessZero.^2)*(lambda/(2*m));
  J=(sum(logErr)/m) + penalty;


  thetaPenalty = thetaAllLessZero.*(lambda/m);
  grad0 = diffErr'*X(:,1)./m;
  gradAllLessZero = diffErr'*X(:,2:end)./m + thetaPenalty';
  gradAll=[grad0,gradAllLessZero];
  grad=gradAll';
  theta=theta-grad;
  % grad = grad(:);

  % Save the cost J in every iteration
  J_history(count,1)=count;
  J_history(count,2)=J;

end
  % plot(J_history(:,1),J_history(:,2),'r+');
  save myGradients.txt grad -ascii;
  save itCanBeDone2.txt J_history -ascii;
  save myThetas.txt theta -ascii;

% =============================================================
end
