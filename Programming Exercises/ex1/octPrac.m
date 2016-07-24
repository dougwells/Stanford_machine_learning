% data = load('ex1data1.txt');
% Z=[1 2; 3 4; 5 6];
% X = [3.*ones(3,3)];
%
% fprintf('x values: \n');
% fprintf('%f \n', data(:,1));
%
% printf("y matrix: \n");
% fprintf('%f \n', y);
% y=Z[:,1]
%
%
% Create new matrix
% for i = 1:10;
%    for j = 1:2;
% 	  test(i,j) = i*10 +j*5;
%    end
% end
%
% 	x=test(:,1);
% 	y=test(:,2);
%
% 	printf("x \n");
% 	fprintf('%f \n', x);
%
% 	plot(x, y, 'rx', 'MarkerSize', 5);

data = load('itCanBeDone.txt');
x=data(:,1);
y=data(:,2);
% printf('y \n');
% fprintf('%f', y);

plot(x, y, 'rx', 'MarkerSize', 5)
