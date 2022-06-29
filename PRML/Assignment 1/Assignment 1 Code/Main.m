%Converting image to double matrix
imRawData = imread('orig.jpg');
imData = double(imRawData);

normEVD = [];
normSVD = [];

K = 20;

for K = 1:17:256
    K
    % Singular Value Decomposition
    W = imData * (imData');
    [USVD, lambda] = eig(W);
    USVD = flip(USVD, 2);
    SSVD = sqrt(lambda);
    VTSVD = inv(SSVD) * inv(USVD) * imData;

    % Image Compression by nullifying everything except the first K Singular Values
    SSVD = SSVD(1:K, 1:K);
    USVD = USVD(:, 1:K);
    VTSVD = VTSVD(1:K, :);

    % Reconstructing the image
    CompImSVD = USVD * SSVD * VTSVD;

    % Displaying the Compressed Image
    figure;
    buffer = sprintf('Image output using %d singular values', K);
    imshow(uint8(CompImSVD));
    imwrite(uint8(CompImSVD), sprintf('%dsvd.jpg', K));
    title(buffer);

    % Eigen Value Decomposition
    [VEVD, SEVD] = eig(imData);

    % Image Compression by nullifying everything except the first K Eigen Values
    CompEVD = SEVD;
    CompEVD(K + 1:end, :) = 0;
    CompEVD(:, K + 1:end) = 0;

    % Reconstructing the image
    CompImEVD = VEVD * CompEVD * inv(VEVD);
    CompImEVD = real(CompImEVD);

    % Displaying the Compressed Image
    figure;
    buffer = sprintf('Image output using %d singular values', K);
    imshow(uint8(CompImEVD));
    imwrite(uint8(CompImEVD), sprintf('%deig.jpg', K));
    title(buffer);
    
    normSVD = [normSVD abs(norm(imData, 'fro') - norm(CompImSVD, 'fro'))];
    normEVD = [normEVD abs(norm(imData, 'fro') - norm(CompImEVD, 'fro'))];
end

plot(1:17:256, normSVD, 'Color', 'r');
title('Comparison');

hold on;
plot(1:17:256,normEVD, 'Color', 'b');

hold off