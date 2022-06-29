data = dlmread('2d_team_34_train.txt');

numTrainEx = 100;
M = 10;
lambda = -5;

y = data(:, 3);
x2 = data(:, 2);
x1 = data(:, 1);

plot3(x1, x2, y, 'ro');
str = sprintf('Degree = %d, No. of Training Examples %d, Lambda = %d', M, numTrainEx, lambda);
title(str);
hold on;

id = randperm(length(data), numTrainEx);

x1 = x1(id);
x2 = x2(id);
y = y(id);

plot3(x1, x2, y, 'bx');

N = length(y);

Y = y;
X = [];


for i = 0:M
    for j = 0:i
        X = [X (x1.^(i-j)).*(x2.^(j))];
    end
end

a = inv(X' * X) * X' * Y;

Y1 = zeros(20, 20);

t1 = [-1.0:0.1:0.9];
t2 = [-1.0:0.1:0.9];

p = 1;

for xax=1:20
    for yax=1:20
        p = 1;
        for i=0:M
            for j=0:i
                Y1(xax, yax) = Y1(xax, yax) + a(p)*(t1(xax)^(i-j))*(t2(yax)^(j));
                p = p+1;
            end
        end
    end
end

[I, J] = meshgrid(-1.0:0.1:0.9,-1.0:0.1:0.9);

surf(t1, t2, Y1);
hold off;