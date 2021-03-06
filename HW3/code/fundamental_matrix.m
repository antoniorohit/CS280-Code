function [F, res] = fundamental_matrix(matches)
% FUNDAMENTAL_MATRIX
% INPUT: matches
% Takes in the matches matrix which is an Nx4 
% matrix containing N corresponding (x1, y1), (x2, y2)
% pairs
%
% OUTPUT: F, res
% Fundamental Matrix (F) and residual error (res)
%
% Ref: 
% https://github.com/jasonzliang/cs280/blob/master/hw2/code/fundamental_matrix.m
% http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf

%% STEP 0
% Calculate the points X1 and X2 (points1 and points2)
% Augment the points with a ones column (homogeneous coordinates?) 
% This column actually helps with creating T1 and T2 (nifty trick
% :))
[N, d] = size(matches);
one_col = ones(N,1);
points1 = [matches(:,[1,2]) one_col]';
points2 = [matches(:,[3,4]) one_col]';

%% STEP 1 NORMALIZATION
% Subtract mean and scale by sigma
mean_matches = mean(matches);
sigma_matches = std(matches);

scaling1 = sqrt(2/(sigma_matches(1)^2+sigma_matches(2)^2));
scaling2 = sqrt(2/(sigma_matches(3)^2+sigma_matches(4)^2));

T1 = scaling1.*[eye(2) -mean_matches(1:2)'];
T1 = [T1; 0 0 1];

T2 = scaling2.*[eye(2) -mean_matches(3:4)'];
T2 = [T2; 0 0 1];

% Homogenous normalized coordinates
pts1 = (T1*points1)';
pts2 = (T2*points2)';

%% STEP 2 OPTIMIZATION
A = ones(N, 9);

% Define A
for i = 1:N
    A(i, 1) = pts1(i, 1)*pts2(i, 1);      % x1*x2
    A(i, 2) = pts1(i, 2)*pts2(i, 1);      % y1*x2
    A(i, 3) = pts2(i, 1);                    % x2
    A(i, 4) = pts1(i, 1)*pts2(i, 2);      % x1*y2
    A(i, 5) = pts1(i, 2)*pts2(i, 2);      % y1*y2
    A(i, 6) = pts2(i, 2);                    % y2
    A(i, 7) = pts1(i, 1);                    % x1
    A(i, 8) = pts1(i, 2);                    % y1
%   A(i, 9) = 1;
end

% SVD where S is eigen value diag matrix
% numeric unitary matrices U and V with the columns 
% containing the singular vectors, and a diagonal matrix 
% S containing the singular values
[U, S, V] = svd(A);

% See equation75 here:
% http://db.cs.duke.edu/courses/cps111/spring07/notes/12.pdf
% get column 9 corresponding to eigen value 9 (smallest)
v9 = V(:,9);

% f is the column vector of elems of F stacked
F_star = [  v9(1), v9(2), v9(3);
            v9(4), v9(5), v9(6);
            v9(7), v9(8), v9(9)];
    
[U, S, V] = svd(F_star);

% Force the rank of F to be 2 (force the smallest eigen value to 0)
S(9) = 0;

F = U*S*V';

%% STEP 3 DENORMALIZATION
F = T2'*F*T1;

%% STEP 4 RESIDUAL ERROR
% See (f) here: 
% http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BEARDSLEY/node2.html
% for how to calculate perpendicular distance of point from line in 
% homogenous coordinates
% Note: we are doing RMS error

% Actual points, not scaled points
X2 = points2';
X1 = points1';


% first residual
eLine1 = X2*F;      % first epipolar line (not yet normalized)
%            dot product                    sqrt(a^2+b^2)
res = sqrt(sum((sum(eLine1.*X1,2)).^2)/N);

end

