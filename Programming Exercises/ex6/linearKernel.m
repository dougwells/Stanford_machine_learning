function sim = linearKernel(x1, x2)
%LINEARKERNEL returns a linear kernel between x1 and x2
%   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Compute the linear kernel.  Note, this can be many different functions
% sim = x1+2 works as does sim = x1.^2 * x2.  So does sim = 1 or 0
% SVM will simply conclude these "kernels" are not predictive and bring their
% corresponding thetas to zero

sim = x1' * x2;  % dot product
% sim = x1 + x2;  % simply define linear "kernel"/similarity as vector 1 + vector 2


end
