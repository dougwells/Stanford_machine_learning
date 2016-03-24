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

% Note: grad should have the same dimensions as theta
%
% !!! Change to the following values when "submitting"
%     alpha = 0.0001;
%     loopend = 1;
alpha=1;
loopEnd=1;
%
% for count=1:loopEnd;
  pred = sigmoid(X*theta);
  act=y;
  err1 = -y.*log(pred);
  err2= (1-y).*log(1-pred);
  diffErr=pred-act;
  logErr = err1-err2;
  thetaZero = theta(1,:);
  thetaAllLessZero=theta(2:end,:);
  penalty = sum(thetaAllLessZero.^2)*(lambda/(2*m));
  grad=diffErr'*X*(alpha/m);
  J=(sum(logErr)/m) + penalty;
  theta=theta-grad';

  % Save the cost J in every iteration
  % J_history(count,1)=count;
  % J_history(count,2)=J;
  % plot(J_history(:,1),J_history(:,2),'r+');

% end
% save itCanBeDone2.txt J_history -ascii;
% save myThetas.txt theta -ascii;




% =============================================================

end
