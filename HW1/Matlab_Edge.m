clc
clear
close
files = dir('*.jpg');

for file = files'
    if(strfind(file.name, 'sol'))
        file.name
    else
        pic = imread(file.name, 'jpg');
        pic_gray = rgb2gray(pic);
        figure()
        imshow(edge(pic_gray,'canny'))
    end
end