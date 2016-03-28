function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



% Setup some useful variables
L = 2;
m = size(X, 1);
K1 = size(Theta1,1)
K2 = size (Theta2,1)

% Add column of 1's to X
X=[ones(m,1) X];

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
for l=1:L
  endK = strcat('K', num2str(l));
  if L==1; Theta = Theta1; inputs = X;end
  if L==2; Theta = Theta2; inputs = L1Output; end

  for k=1:endK
    currTheta = Theta(k,:)';
    pred = sigmoid(inputs*currTheta);
    act=(y==k);
    err1 = -act.*log(pred);
    err2= (1-act).*log(1-pred);
    diffErr=pred-act;
    logErr = err1-err2;
    thetaZero = currTheta(1,:);
    thetaAllLessZero=currTheta(2:end,:);
    % set costPenalty & thetaPenalty= 0 for part 1 of Wk5 Exercise
    % costPenalty = sum(thetaAllLessZero.^2)*(lambda/(2*m));
    costPenalty = 0;
    J=(sum(logErr)/m) + penalty;
  end
end
  % % thetaPenalty = thetaAllLessZero.*(lambda/m);
  % thetaPenalty=0;
  % grad0 = diffErr'*X(:,1)./m;
  % gradAllLessZero = diffErr'*X(:,2:end)./m + thetaPenalty';
  % gradAll=[grad0,gradAllLessZero];
  % grad=gradAll';
  % theta=theta-grad;
  % % grad = grad(:);
  %
  % % Save the cost J in every iteration
  % J_history(count,1)=count;
  % J_history(count,2)=J;
  %
  % % plot(J_history(:,1),J_history(:,2),'r+');
  % save myGradients.txt grad -ascii;
  % save itCanBeDone2.txt J_history -ascii;
  % save myThetas.txt theta -ascii;








%Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
