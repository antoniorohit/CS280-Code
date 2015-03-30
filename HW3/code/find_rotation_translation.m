function [R, t] = find_rotation_translation(E)
%FIND_ROTATION_TRANSLATION Summary of this function goes here
%   Detailed explanation goes here

% See: 
% https://www8.cs.umu.se/kurser/TDBD19/VT05/epipolar-4.pdf


[U, S, V] = svd(E);

W = [0 -1  0;
     1  0  0;
     0  0  1];
 
%% Translation Matrix
t = cell(2,1); 
t{1} = +U(:,end);
t{2} = -t{1};

%% Rotation Matrix
% Do we have to check whether the determinant of R is 1?

R = cell(2,1);
R{1} = U*W*V';
detR1 = (det(R{1}));

R{2} = U*W'*V';
detR2 = (det(R{2}));

if (abs(detR1-1) > 0.001) || (abs(detR2-1) > 0.001)
    fprintf('Error: Determinant of the rotation matrix is R1=%f, R2=%f\n', detR1, detR2);
    % What to do
end

end

