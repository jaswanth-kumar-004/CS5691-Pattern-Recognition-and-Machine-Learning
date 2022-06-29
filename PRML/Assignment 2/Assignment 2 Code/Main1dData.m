%READING TRAINING DATA
data = dlmread('1d_team_34_train.txt');

trainingError = zeros(25, 1);
numTrainExList = [10; 20; 40; 80; 160];
numTrainEx = 80;                            % Number of Training Examples
M = 10;                                     % Degree of Polynomial
lambda = -5;                                % Lambda
k = 1;
for M = 5:5:25
    for num = 1:5
        numTrainEx = numTrainExList(num);
        y = data(:, 2);                             % Output
        x = data(:, 1);                             % Input

        plot(x, y, 'ro');                           % Plotting all data points
        str = sprintf('Degree = %d, No. of Training Examples %d, Lambda = %d', M, numTrainEx, lambda);
        title(str);
        hold on;

        id = randperm(length(data), numTrainEx);

        x = x(id);
        y = y(id);

        plot(x, y, 'bx');                           % Plotting the uniformly sampled data points 

        N = length(y);

        Y = y;
        X = [];

        for i = 0:1:M
            X = [X x.^(i)];                         % Basis Function
        end

        a = inv(X' * X + lambda * eye(M + 1)) * X' * Y; % Findong the Weight Vector

        Y1 = zeros(N, 1);

        t = [0:0.1:4.9];

        for i = 0:M
            Y1 = Y1 + a(i+1) * t.^(i);
        end
        
        plot(t, Y1, '-r');
        figure;
        hold off;
        
        Y2 = zeros(N, 1);
        
        for i = 0:M
            Y2 = Y2 + a(i+1) * Y.^(i);
        end

        trainError(k) = sum((y - Y2).^2)/N;             % Training Error
        k = k+1;
    end
end