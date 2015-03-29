function [  ] = plot_3d(points, P1, P2)
%PLOT_3D Summary of this function goes here
%   Detailed explanation goes here

X1 = points(:,1);
Y1 = points(:,2);
Z1 = points(:,3);

figure();
% 3D pointss
plot3(X1, Y1, Z1, 'R.');
hold on;

% Camera Centres
plot3(P1(1,4), P1(2,4), P1(3,4), 'B+');
plot3(P2(1,4), P2(2,4), P2(3,4), 'B+');

end

