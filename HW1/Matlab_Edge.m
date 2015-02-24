clc
clear
close
files = dir('*.jpg');

for file = files'
    pic = imread(file.name, 'jpg');
    pic_gray = rgb2gray(pic);
    figure()
    imshow(edge(pic_gray,'canny', .4, 1))
end