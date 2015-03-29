function [points, rec_err] = find_3d_points(P1, P2, matches)
%FIND_3D_POINTS Summary of this function goes here
%   Detailed explanation goes here
    
%%Find the 3D points
[N, d] = size(matches);

points = zeros(N, 3);

% loop over all points in matches
for i = 1:N
    % Using the equation given in the HW2 pdf, applying W=1,
    % multiplying by the denominator and collecting terms
    % We get the equation in the form AX = b
    % Where X = [X Y Z]
    x1 = matches(i, 1);
    y1 = matches(i, 2);
    x2 = matches(i, 3);
    y2 = matches(i, 4);

    A = [(x1*P1(3,1)-P1(1,1)) (x1*P1(3,2)-P1(1,2)) (x1*P1(3,3)-P1(1,3));
         (y1*P1(3,1)-P1(2,1)) (y1*P1(3,2)-P1(2,2)) (y1*P1(3,3)-P1(2,3));
         (x2*P2(3,1)-P2(1,1)) (x2*P2(3,2)-P2(1,2)) (x2*P2(3,3)-P2(1,3));
         (y2*P2(3,1)-P2(2,1)) (y2*P2(3,2)-P2(2,2)) (y2*P2(3,3)-P2(2,3));    
        ];

    % Since W=1, we take 
    b = -[(x1*P1(3,4)-P1(1,4));
          (y1*P1(3,4)-P1(2,4));
          (x2*P2(3,4)-P2(1,4));
          (y2*P2(3,4)-P2(2,4));
        ];

    % Least squares SVD (W=1)
    [U, S, V] = svd(A);

    % below eq 72:
    % http://db.cs.duke.edu/courses/cps111/spring07/notes/12.pdf
    % fix the dimensions of S^-1
    S_adjusted = [S((1:3),:)^-1 zeros(3,1)];

    % eq 72:
    points(i,:) = V*S_adjusted*U'*b;
end

%% Reconstruction error
% x1=P1*X, x2=P2*X
X1 = (P1*[points ones(N, 1)]')';
X2 = (P2*[points ones(N, 1)]')';

% Adjust for the homogenous coordinates
for i=1:N
    X1(i,:)=X1(i,:)./X1(i,3);
    X2(i,:)=X2(i,:)./X2(i,3);
end

% The outer sum is over x and y, then inner sum is over all the N elements
% Divided by 2*N because error is calculated for each image
% sqrt(sum(sum((X3D-X2D)^2 + (Y3D-Y2D)^2)))/N   <- for each image, then
% take average
rec_err = (sqrt(sum(sum((X1(:,[1,2])-matches(:,[1,2])).^2))) + ...
           sqrt(sum(sum((X2(:,[1,2])-matches(:,[3,4])).^2))))/(2*N);
end

