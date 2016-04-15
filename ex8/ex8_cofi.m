%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
%
% fprintf('Loading movie ratings dataset.\n\n');

%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
% fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
%         mean(Y(1, R(1, :))));
%
% %  We can "visualize" the ratings matrix by plotting it with imagesc
% imagesc(Y);
% ylabel('Movies');
% xlabel('Users');
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);

% fprintf(['Cost at loaded parameters: %f '...
%          '\n(this value should be about 22.22)\n'], J);
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement
%  the collaborative filtering gradient function. Specifically, you should
%  complete the code in cofiCostFunc.m to return the grad argument.
%
% fprintf('\nChecking Gradients (without regularization) ... \n');
%
% %  Check gradients by running checkNNGradients
% checkCostFunction;
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%

%  Evaluate cost function
% J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
%                num_features, 1.5);
%
% fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
%          '\n(this value should be about 31.34)\n'], J);
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement
%  regularization for the gradient.
%

%
% fprintf('\nChecking Gradients (with regularization) ... \n');
%
% %  Check gradients by running checkNNGradients
% checkCostFunction(1.5);
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
% my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
% my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:

% % David's Film Ratings
my_ratings(1) = 5;
my_ratings(2) = 3;
my_ratings(3) = 4;
my_ratings(4) = 4;
my_ratings(7) = 5;
my_ratings(8) = 5;
my_ratings(11) = 5;
my_ratings(12) = 5;
my_ratings(13) = 3;
my_ratings(14) = 4;
my_ratings(15) = 3;
my_ratings(17) = 3;
my_ratings(22) = 4;
my_ratings(23) = 5;
my_ratings(25) = 5;
my_ratings(27) = 2;
my_ratings(28) = 4;
my_ratings(29) = 2;
my_ratings(31) = 3;
my_ratings(33) = 4;
my_ratings(35) = 2;
my_ratings(38) = 2;
my_ratings(39) = 5;
my_ratings(41) = 3;
my_ratings(42) = 5;
my_ratings(47) = 4;
my_ratings(49) = 4;
my_ratings(50) = 5;
my_ratings(53) = 2;
my_ratings(54) = 3;
my_ratings(55) = 5;
my_ratings(56) = 5;
my_ratings(58) = 5;
my_ratings(59) = 5;
my_ratings(60) = 5;
my_ratings(61) = 5;
my_ratings(62) = 4;
my_ratings(63) = 3;
my_ratings(64) = 5;
my_ratings(66) = 4;
my_ratings(67) = 3;
my_ratings(69) = 4;
my_ratings(70) = 5;
my_ratings(71) = 4;
my_ratings(72) = 3;
my_ratings(73) = 3;
my_ratings(76) = 4;
my_ratings(77) = 3;
my_ratings(78) = 3;
my_ratings(79) = 4;
my_ratings(80) = 2;
my_ratings(81) = 5;
my_ratings(82) = 5;
my_ratings(83) = 5;
my_ratings(86) = 5;
my_ratings(87) = 4;
my_ratings(88) = 4;
my_ratings(89) = 5;
my_ratings(90) = 3;
my_ratings(92) = 4;
my_ratings(94) = 3;
my_ratings(95) = 4;
my_ratings(96) = 3;
my_ratings(97) = 5;
my_ratings(98) = 5;
my_ratings(99) = 5;
my_ratings(100) = 5;


% % Hanna's Film Ratings
% my_ratings(2) = 4;
% my_ratings(22) = 2;
% my_ratings(41) = 4;
% my_ratings(50) = 5;
% my_ratings(71) = 5;
% my_ratings(91) = 5;
% my_ratings(82) = 5;
% my_ratings(67) = 2;
% my_ratings(89) = 3;

% % Carrie's Film Ratings
% my_ratings(1) = 4;
% my_ratings(7) = 2;
% my_ratings(11) = 3;
% my_ratings(15) = 4;
% my_ratings(28) = 4;
% my_ratings(31) = 4;
% my_ratings(35) = 1;
% my_ratings(40) = 1;
% my_ratings(41) = 2;
% my_ratings(50) = 4;
% my_ratings(54) = 3;
% my_ratings(56) = 4;
% my_ratings(64) = 4;
% my_ratings(69) = 1;
% my_ratings(82) = 4;
% my_ratings(87) = 4;
% my_ratings(91) = 4;
% my_ratings(95) = 4;
% my_ratings(98) = 5;
% my_ratings(100) = 3;

% % Mason's Film Ratings
% my_ratings(1) = 5;
% my_ratings(2) = 3;
% my_ratings(12) = 5;
% my_ratings(29) = 4;
% my_ratings(53) = 2;
% my_ratings(56) = 2;
% my_ratings(79) = 4;
% my_ratings(94) = 4;
% my_ratings(97) = 5;
% my_ratings(98) = 5;

% % Wayne's Film Ratings
% my_ratings(7) = 4;
% my_ratings(81) = 4;
% my_ratings(34) = 2;
% my_ratings(64) = 4;
% my_ratings(43) = 3;
% my_ratings(1) = 4;
% my_ratings(12) = 4;
% my_ratings(35) = 1;
% my_ratings(89) = 4;
% my_ratings(73) = 3;

% % Devin's Film Ratings
% my_ratings(1) = 4;
% my_ratings(12) = 4;
% my_ratings(17) = 3;
% my_ratings(22) = 5;
% my_ratings(41) = 4;
% my_ratings(50) = 5;
% my_ratings(51) = 2;
% my_ratings(64) = 5;
% my_ratings(70) = 4;
% my_ratings(85) = 4;
% my_ratings(98) = 5;

% % Jason's film Ratings
% my_ratings(97) = 4;
% my_ratings(2) = 4;
% my_ratings(28) = 5;
% my_ratings(51) = 5;
% my_ratings(24) = 1;
% my_ratings(38) = 2;
% my_ratings(68) = 3;
% my_ratings(77) = 4;
% my_ratings(27) = 2;
% my_ratings(43) = 3;

% % Doug Wells Ratings
% my_ratings(1) = 5;
% my_ratings(8) = 3;
% my_ratings(11)= 4;
% my_ratings(15) = 5;
% my_ratings(28)= 4;
% my_ratings(35)= 2;
% my_ratings(48) = 4;
% my_ratings(71) = 5;
% my_ratings(77) = 4;
% my_ratings(82)= 4;
% my_ratings(98) = 4;
% my_ratings(99)= 4;
% my_ratings(117)= 3;
% my_ratings(114)= 2;
% my_ratings(121)= 4;
% my_ratings(122)= 4;
% my_ratings(127)= 5;
% my_ratings(131)= 4;
% my_ratings(139)= 4;
% my_ratings(154)= 2;
% my_ratings(161)= 4;
% my_ratings(168)= 2;
% my_ratings(174)= 5;
% my_ratings(186)= 2;
% my_ratings(201)= 0;
% my_ratings(219)= 1;
% my_ratings(234)= 1;

% fprintf('\n\nNew user ratings:\n');
% for i = 1:length(my_ratings)
%     if my_ratings(i) > 0
%         fprintf('Rated %d for %s\n', my_ratings(i), ...
%                  movieList{i});
%     end
% end

% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 500);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

% fprintf('Recommender system learning completed.\n');
%
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%


p = X * Theta';
my_predictions = p(:,1) + Ymean;

% Share and save data on model
size(my_ratings)
size(X)
size(Theta)
size(my_predictions)
save moviesThetas.txt Theta -ascii;
save moviesX.txt X -ascii;
save my_predictions.txt my_predictions -ascii;

% Load Movie List
movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');

% Printing out top 20 films recommended (DW subtracted 3 so on a 5 star scale)
fprintf('\nTop recommendations for you:\n');
for i=1:50
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

my_predictions(1,:)
movieList{1,:}

% Save movie recommendations
movieRatings = {};
for i=1:1500
    j = ix(i);
    movieRatings{i,1} = my_predictions(j,1);
    movieRatings{i,2} = movieList(j,:);
end
save movieRatingsForCarrie.txt movieRatings -ascii;
