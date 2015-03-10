clear;
close;
clc;

% TRAIN
% Size of scene
M = 11;
scale = 1;
% proportion of road
k_road = 0.4;

train_wall = 1000; % metres
train_scene = train_wall*ones(M, M);
ground_dist = 1000;
plane_scene = ground_dist*ones(M, M);
train_vel = 1000000; % m/s

car_vel = 1000000;
car_ht = 2;

plane_y_vel = 1000000;
plane_z_vel = 1000000;


% fill in the road
for i = 1:M*k_road
    for j = 1:M
        z = (i)*(1000/(M*k_road));
        train_scene(i, j) = z;
    end
end

u_train = zeros(size(train_scene)/scale);
v_train = zeros(size(train_scene)/scale);

u_car = zeros(size(train_scene));
v_car = zeros(size(train_scene));

u_plane = zeros(size(plane_scene));
v_plane = zeros(size(plane_scene));

u_rot = zeros(size(plane_scene));
v_rot = zeros(size(plane_scene));

% TRAIN
u_train = (-train_vel./train_scene());

% CAR
for i = 1:M
    for j = 1:M
                v_car(i,j) = (car_vel*(i-M*k_road)/train_scene(i,j));
                u_car(i,j) = (car_vel*(j-M/2)/train_scene(i,j));
    end
end

% PLANE
for i = 1:M
    for j = 1:M
            v_plane(i,j) = (-plane_y_vel+plane_z_vel*(i-M*0.5))/plane_scene(i,j);
            u_plane(i,j) = (plane_z_vel*(j-M*0.5)/plane_scene(i,j));
    end
end

w = 10; %wx = wy

% ROTATION
for i = 1:M
    for j = 1:M
            v_rot(i,j) = -(1+(j-M/2)^2)*w*0.5 + (i-M/2)*(j-M/2)*w*0.5;
            u_rot(i,j) = (1+(i-M/2)^2)*w*0.5 - (i-M/2)*(j-M/2)*w*0.5;
    end
end

hold on;
ha1 = area([0 M+1], [M M]);
set(ha1,'FaceColor',[.5 .5 .5]);
set(ha1,'EdgeColor','None');

ha2 = area([0 M+1], [M*k_road M*k_road]);
set(ha2,'FaceColor',[1 1 1]);
set(ha2,'EdgeColor','None');

saveas(quiver(u_train,v_train), 'Optical_Flow_Train.jpg');
axis([0 M+1 0 M+1]);
title('Optical Flow for Train');

figure()
hold on;
ha3 = area([0 M+1], [M M]);
set(ha3,'FaceColor',[.5 .5 .5]);
set(ha3,'EdgeColor','None');

ha4 = area([0 M+1], [M*k_road M*k_road]);
set(ha4,'FaceColor',[1 1 1]);
set(ha4,'EdgeColor','None');

saveas(quiver(u_car,v_car), 'Optical_Flow_Car.jpg');
title('Optical Flow for Car');
axis([0 M+1 0 M+1]);

figure()
hold on;
ha3 = area([1 M], [M M]);
set(ha3,'FaceColor',[.5 .5 .5]);
set(ha3,'EdgeColor','None');

ha4 = area([0 M+1], [1 1]);
set(ha4,'FaceColor',[1 1 1]);
set(ha4,'EdgeColor','None');

saveas(quiver(u_plane,v_plane), 'Optical_Flow_Plane.jpg');
title('Optical Flow for Plane');
axis([0 M+1 0 M+1]);

figure()

saveas(quiver(u_rot,v_rot), 'Optical_Flow_Rotation.jpg');
title('Optical Flow for Rotation');
axis([0 M+1 0 M+1]);