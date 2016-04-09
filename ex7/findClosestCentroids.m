function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%



% Set K
K = size(centroids, 1);
m=size(X,1);

% Inspect Data
  % size(X)
  % size(centroids)
  % X(1:3,:)
  % centroids

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:m
    min = 1000;
    for j = 1:K
      % Two ways to calculate Euclidean distance between vectors.  Both Work

        % Method 1:  Use Matlab built-in fn "norm"
          dist = norm(X(i,:)-centroids(j,:)).^2;

        % Method 2:  Calculate Euclidean Distance Ourselves
          % diff = X(i,:)-centroids(j,:);
          % dist = sum(diff.^2);

        % Method 3 Does NOT work since only calculates if vectors have 2 features or less
          % dist = ((X(i,1)-centroids(j,1)).^2 + (X(i,2)-centroids(j,2)).^2 );

        Distance(i,j) = dist;
        if dist < min
          idx(i)=j;
          min = dist;
        end
    end
end
% =============================================================
end
