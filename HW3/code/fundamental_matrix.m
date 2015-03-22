function [F, res] = fundamental_matrix(matches)
% FUNDAMENTAL_MATRIX
% INPUT: matches
% Takes in the matches matrix which is an Nx4 
% matrix containing N corresponding (x1, y1), (x2, y2)
% pairs
%
% OUTPUT: F, res
% Fundamental Matrix (F) and residual error (res)
[N, d] = size(matches);

A = ones(N, 9);

% Define A
for i = 1:N
    A(i, 1) = matches(i, 1)*matches(i, 3);      % x1*x2
    A(i, 2) = matches(i, 2)*matches(i, 3);      % y1*x2
    A(i, 3) = matches(i, 3);                    % x2
    A(i, 4) = matches(i, 1)*matches(i, 4);      % x1*y2
    A(i, 5) = matches(i, 2)*matches(i, 4);      % y1*y2
    A(i, 6) = matches(i, 4);                    % y2
    A(i, 7) = matches(i, 1);                    % x1
    A(i, 8) = matches(i, 2);                    % y1
%   A(i, 9) = 1;
end

% SVD where S is eigen value diag matrix
% numeric unitary matrices U and V with the columns 
% containing the singular vectors, and a diagonal matrix 
% S containing the singular values
[U, S, V] = svd(A);

size(U)
size(S)
size(V)

    
F = [];
res = [];

end

