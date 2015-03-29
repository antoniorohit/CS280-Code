function [R, t] = find_rotation_translation(E)
%FIND_ROTATION_TRANSLATION Summary of this function goes here
%   Detailed explanation goes here

% See: 
% https://www8.cs.umu.se/kurser/TDBD19/VT05/epipolar-4.pdf


[U, S, V] = svd(E);

W = [0 -1  0;
     1  0  0;
     0  0  1];

t = cell(2,1); 
t{1} = +U(:,end);
t{2} = -t{1};
 
R = cell(2,1);
R{1} = U*W*V';
R{2} = U*W'*V';

end

