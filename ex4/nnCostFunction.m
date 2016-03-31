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

% Check sizes of matrices
% size(X)
% size(y)
% size(Theta1)
% size(Theta2)
% y(995:1005,1)'



% Setup some useful variables
L = 2;
m = size(X, 1);
K1 = size(Theta1,1);
K2 = size (Theta2,1);

% Add column of 1's to X
X=[ones(m,1) X];
size(X);
size(Theta1);
size(Theta2);

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

% Compute theta costPenalty for Theta1.  Layer 1 has 400 units (20x20 pixels)
% There is 1 row of thetas per unit in that layer.  a1=Output = g(X*Theta1)
% In this case, there are 400 units in Layer 1 and 25 units in Layer 2.
% Each of the L2's 25 units has 400 inputs plus the bias.
% Note: We don't include bias/1s in thetaPenalty


% Compute a's ("activations").      = g(z) where z=X*Theta'
  a1 = inputsOfL1 = X;              %Bias unit already included
  z2 = a1*Theta1';
  a2 = outputsOfL1 = sigmoid(z2);   % these are the a superscript 2s
  a2wBias = inputsOfL2 = [ones(m,1) a2];       % adding the bias unit
  z3 = a2wBias*Theta2';
  a3 = outputsOfL2 = sigmoid(z3);

% Output so 1 column per label
  for k=1:K2
    Y(:,k) = (y==k);
  end

% Compare outputs of L2 to actual results.
  act=Y;
  pred=a3;
  currTheta = Theta2(k,:)';

% Calculate costs for each activation w/its parameters/weights/thetas
  err1 = -act.*log(pred);
  err2= (1-act).*log(1-pred);
  diffErr=pred-act;
  logErr = err1-err2;
  theta1AllLessZero=Theta1'(2:end,:);
  costPenaltyTheta1 = sum(theta1AllLessZero.^2)*(lambda/(2*m));
  theta2AllLessZero=Theta2'(2:end,:);
  costPenaltyTheta2 = sum(theta2AllLessZero.^2)*(lambda/(2*m));

% "Comment out" cost penalty here.  Will add in down below
  J=(sum(logErr)/m); %+ costPenalty;

% Summarize total costs (1 per activation) + add theta Penalties
  costPenaltyTheta1 = sum(costPenaltyTheta1);
  costPenaltyTheta2 = sum(costPenaltyTheta2);
  J=sum(J)+costPenaltyTheta1 + costPenaltyTheta2;







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

% For backpropagation, we want Theta's w/o bias
    Theta1LessBias = Theta1(:,2:end);
    Theta2LessBias = Theta2(:,2:end);

% Dimension Check
  % z2 = a1*Theta1'
    size(a1);                 % --> 5000 x 401
    size(Theta1LessBias);     % --> 25 x 400
    size(z2);                 % --> 5000 x 25

  % z3 = a2*Theta2'
    size(a2wBias);            % --> 5000 x 26
    size(a2);                 % --> 5000 x 25
    size(Theta2LessBias);     % --> 10 x 25
    size(z3);                 % --> 5000 x 10


% d3 is simply predicted less actual (not Y = 5,000x10)
  d3 = a3-Y;

% Calculate d2
  d2=d3*Theta2LessBias;
  d2 = d2.*sigmoidGradient(z2);  % elementwise multiply d2 by sigmoidGradient

% Dimension Check
  size(d2); % --> 5000 x 25
  size(d3); % --> 5000 x 10

% Compute Delta1 and Delta2 (gradients for Theta1 & Theta2).
% Want to use a's WITH bias units

  Delta1 = d2'*a1;
  size(Delta1); % --> 25 x 401

  Delta2 = d3'*a2wBias;
  size(Delta2);  % --> 10 x 26

  Theta1_grad = Delta1./m;  % --> 25x401
  Theta2_grad = Delta2./m;  % --> 10x26

% Regularize the Gradients (penalize with lambda)

  % Modify Theta1 & Theta2 to make easier to regularize (make first column of each = 0)
    Theta1(:,1)=0;  %(first column is all 0s)
    Theta2(:,1)=0;  %(first column is all 0s)
    size(Theta1);  % --> 25x401
    size(Theta2);  % --> 10x26

  % Scale Theta1 & Theta2 by lamda/m
    regularizationTheta1 = Theta1*(lambda/(m));
    regularizationTheta2 = Theta2*(lambda/(m));

  % Add regularized Theta matrices to unregularized Theta Gradients
    Theta1_grad = Theta1_grad + regularizationTheta1;
    Theta2_grad = Theta2_grad + regularizationTheta2;
    size(Theta1_grad); % --> 25x401
    size(Theta2_grad); % --> 10x26

% % Initialize random thetas as a "guess".  (L_in, L_out) --> matrix(L_out,L_in+1)
% Theta1 = randInitializeWeights(400,25);   % --> (25 x 401)
% Theta2 = randInitializeWeights(25,10);    % --> (10 x 26)
%
% %Forward Propogation.  NN output based on first training example
% % & initial random thetas
%
% for t=1:1  %(t = 1 to m.  One for each training sample)
%   %Calculate a2's & a3's
%   a1(t,:) = inputsOfL1(t,:) = X(t,:);
%   size(a1)
%   size(Theta1')
%   a2(t,:)= outputsOfL1(t,:) = sigmoid(inputsOfL1(t,:)*Theta1');  % these are the a superscript 2s (1x25)
%   size(a2)
%   inputsOfL2(t,:) = [ones(size(outputsOfL1(t,:),1),1) outputsOfL1(t,:)];  % adding the bias unit
%   a3(t,:) = outputsOfL2(t,:) =predicted(t,:)= sigmoid(inputsOfL2(t,:)*Theta2')';
%   size(predicted)
%
%   for j=1:K2    %K2 = # of output classes (is this a 1 or not?  is this a 2 or not, etc)
%     act = y(t,:)==k;
%     d3(j,:)=predicted(k,:)-act;
%   end
%
% % Calculate d2 using for-loop
% % K1 = number of units in Layer 2/first hidden layer.
% % i+1 --> to Bypass bias unit
%   for i=1:K1
%     for j=1:K2
%       dTemp(i,j)=Theta2(j,i+1)*d3(j,:);
%     end
%     d2(i,:) = sum(dTemp(i,:));
%   end
%   % size(d2)
%
% % Calculate d2 using an array (1 row per unit at that level --> (#units x 1))
%   dArray=Theta2'*d3;
%   d2Array = dArray(2:end,:);
%
%
% % add g-prime (sigmoidGradient) to d2Array
% % note:  I am keeping a2 format same as X (each test case is a row)
% % need to transpose a2 as each test case's input is actually its columns
%   d2 = d2.*a2'.*(1-a2');
%
% Delta=0;
% % for i=1:K1
% %   for j=1:K2
% %     Delta2(i,j) = a2(i,j)*d3(j);
% %     % Delta2(i,j) = d3(j,:)*a2(i,j)'
% %   end
% % end
% % size(Delta2)
%
% Delta=d3*a2;
% size(Delta)
%
%
%
% end







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
