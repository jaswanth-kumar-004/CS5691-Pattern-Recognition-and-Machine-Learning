clear all;
clc;

% Parameters
K = 10;
epoch = 100;

color = ['.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'];
% Data Extraction for Synthetic Data
trainData = dlmread('34/train.txt');
devData = dlmread('34/dev.txt');

%Variabe Creation
clTrain = cell(1, 5);
clTrain{1} = trainData(1:1250, 1:end-1);
clTrain{2} = trainData(1251:2500, 1:end-1);
d = size(clTrain{1}, 2);

kval = [2:2:20]';
error = zeros(size(kval, 1), 1);
confMat = zeros(2, 2, size(kval, 1));
f1 = zeros(size(kval, 1), 1);

for ks = 1:size(kval, 1)
    ks
    K = kval(ks);
    cent = cell(1, 5);
    cov = cell(1, 5);
    r = cell(1, 5);
    prevr = cell(1, 5);

    for i = 1:d
        r{i} = zeros(size(clTrain{i}(:, 1), 1), K);
    end

    % Start
    figure;
    scatter(clTrain{1}(:, 1), clTrain{1}(:, 2), 'filled');
    hold on;
    scatter(clTrain{2}(:, 1), clTrain{2}(:, 2), 'filled');
    title('Training Data Scatter Plot')
    hold off;

    figure;
    hold on;

    for clNum = 1:d
        randIndex = randsample(1:size(clTrain{clNum}(:, 1)), K);
        cent{clNum} = (clTrain{clNum}(randIndex, :));

        for iter = 1:epoch
            for i = 1:size(clTrain{clNum}(:, 1))
                for j = 1:size(cent{clNum}(:, 1))
                    r{clNum}(i, j) = norm(clTrain{clNum}(i, :) - cent{clNum}(j, :));
                end
                [minTemp, I] = min(r{clNum}(i, :));
                for k = 1:size(cent{clNum})
                    if k == I
                        r{clNum}(i, k) = 1;
                    else
                        r{clNum}(i, k) = 0;
                    end        
                end
            end

            % Find New Mean
            for i = 1:size(cent{clNum})
                cent{clNum}(i, :) = sum(clTrain{clNum} .* r{clNum}(:, i))/sum(r{clNum}(:, i));
            end

            %Convergence criterion
            if isequal(prevr{clNum}, r{clNum})
                break;
            else
                prevr = r;
            end
        end
        color = ['.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'; '.r'; '.g'; '.c'; '.m'; '.y'; '.k'];
        axis([-15 15 -15 15]);
        for i = 1:size(clTrain{clNum})
            [maxTemp, I] = max(r{clNum}(i, :));
            plot(clTrain{clNum}(i, 1), clTrain{clNum}(i, 2), color(I, :), 'MarkerSize', 20);
        end

        cov{clNum} = zeros(2, 2, K);

        for i = 1:K
            for j = 1:size(clTrain{clNum})
                for p = 1:d
                    for q = 1:d
                        cov{clNum}(p, q, i) = cov{clNum}(p, q, i) + r{clNum}(j, i) * (clTrain{clNum}(j, p) - cent{clNum}(i, p)) * (clTrain{clNum}(j, q) - cent{clNum}(i, q));
                    end
                end
            end
        end

        for i = 1:K
            cov{clNum}(:, :, i) = cov{clNum}(:, :, i)./sum(r{clNum}(:, i));
            G = gmdistribution(cent{clNum}(i, :), cov{clNum}(:, :, i));
            F = @(x,y) pdf(G,[x y]);
            fcontour(F);
        end
    end

    error(ks) = 0;

    for clNum = 1:d
        for i = 1:size(clTrain{clNum}, 1)
            for j = 1:size(cent{clNum}, 1)
                error(ks) = error(ks) + r{clNum}(i, j) * norm((clTrain{clNum}(i, :) - cent{clNum}(j, :)))^2;
            end
        end
    end
    
    % Confusion Matrix
    centroids = [];
    for clNum = 1:d
        centroids = [centroids; [cent{clNum} ones(size(cent{clNum}, 1), 1)* clNum]];
    end
    
    devData = [devData zeros(size(devData, 1), 1)];
    
    for i = 1:size(devData)
        dist = Inf;
        for j = 1:size(centroids)
            if dist > norm(devData(i, 1:2) - centroids(j, 1:2))
                dist = norm(devData(i, 1:2) - centroids(j, 1:2));
                devData(i, 4) = centroids(j, 3);
            end
        end
    end
    
    for i = 1:size(devData, 1)
        if devData(i, 4) == 1 && devData(i, 3) == 1
            confMat(1, 1, ks) = confMat(1, 1, ks) + 1;
        end
        if devData(i, 4) == 1 && devData(i, 3) == 2
            confMat(1, 2, ks) = confMat(1, 2, ks) + 1;
        end
        if devData(i, 4) == 2 && devData(i, 3) == 1
            confMat(2, 1, ks) = confMat(2, 1, ks) + 1;
        end
        if devData(i, 4) == 2 && devData(i, 3) == 2
            confMat(2, 2, ks) = confMat(2, 2, ks) + 1;
        end
    end
    precision = confMat(1, 1, ks)/(confMat(1, 1, ks) + confMat(1, 2, ks));
    recall    = confMat(1, 1, ks)/(confMat(1, 1, ks) + confMat(2, 1, ks));
    f1(ks)      = 2 * precision * recall / (precision + recall);
    
    x = linspace(-15, 15, 100);
    y = linspace(-15, 15, 100);
    
    devPlot = zeros(10);
    
    for a = 1:size(x, 2)
        for b = 1:size(y, 2)
            dist = Inf;
            for j = 1:size(centroids)
                if dist > norm([x(a) y(b)] - centroids(j, 1:2))
                    dist = norm([x(a) y(b)] - centroids(j, 1:2));
                    devPlot(a, b) = centroids(j, 3);
                end
            end
        end
    end
    
    color = ['r.'; 'b.'];
    figure;
    title('Training Data Scatter Plot');
    for i = 1:size(x, 2)
        for j = 1:size(y, 2)
            plot(x(i), y(j), color(devPlot(i, j), :), 'MarkerSize',12);
            hold on;
        end
    end
    scatter(clTrain{1}(:, 1), clTrain{1}(:, 2), 'filled');
    scatter(clTrain{2}(:, 1), clTrain{2}(:, 2), 'filled');
end

figure;
plot(kval(1:size(error)), error);
hold off;

figure;
plot(kval(1:size(f1)), f1);
hold off;

% K = 12 is the optimal value observing from the error graph and the f1
% score graph