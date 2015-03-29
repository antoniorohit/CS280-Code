function [  ] = plot_3d( points )
%PLOT_3D Summary of this function goes here
%   Detailed explanation goes here

X1 = points(:,1);
Y1 = points(:,2);
Z1 = points(:,3);

figure();
% use plot3 command
plot3(X1, Y1, Z1, 'R.');

end

