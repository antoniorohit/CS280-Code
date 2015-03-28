function [R t] = find_rotation_translation(E)
%FIND_ROTATION_TRANSLATION Summary of this function goes here
%   Detailed explanation goes here

% See: 
% https://www8.cs.umu.se/kurser/TDBD19/VT05/epipolar-4.pdf


[U, S, V] = svd(E);

W = [0 -1  0;
     1  0  0;
     0  0  1];

% One way
 
 R = U*W*V';

% Alternative?
R2 = U*W'*V'


t = E*R^-1;

end

