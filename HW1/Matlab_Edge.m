clc
clear
close
pic = imread('pic1.jpg', 'jpg');
pic_gray = rgb2gray(pic);
%imshow(pic_gray)
imshow(edge(pic_gray,'canny', .4, 1))