pwd()

data_dir = ['../data/house'];

% images
I1 = imread([data_dir '/' 'house' '1.jpg']);
I2 = imread([data_dir '/' 'house' '2.jpg']);

imshow(I1);
figure()
imshow(I2);

a = reconstruct_3d('house')
