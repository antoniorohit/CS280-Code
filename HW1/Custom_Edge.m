clc
clear
close

files = dir('*.jpg');

for file = files'
    pic = imread(file.name, 'jpg');
    pic_gray = rgb2gray(pic);
    % STEP 1
    G = fspecial('gaussian',[5, 5], 2);
    pic_conv = (imfilter(pic_gray, G, 'same'));

    % STEP 2
    dx = [-1/2, 0, 1/2];
    Gx = conv2(pic_conv, dx, 'same');

    dy = dx';
    Gy = conv2(pic_conv, dy, 'same');

    % Magnitude of gradient
    Im = sqrt(Gx.*Gx + Gy.*Gy);

    % Angle of gradient
    theta = round(atan2(Gy, Gx)*180/pi);


    % STEP 3
    mark = zeros(size(pic_conv));
    edge = zeros(size(pic_conv));
    [m, n] = size (pic_conv);
    startline_thresh = 10;
    stopline_thresh = 2;

    for i = 2: m-1
        for j = 2 : n-1
            if mark(i,j) == 0
                n1 = Im(i-1, j-1);
                n2 = Im(i-1, j  );
                n3 = Im(i-1, j+1);
                n4 = Im(i  , j-1);
                n5 = Im(i  , j+1);
                n6 = Im(i+1, j-1);
                n7 = Im(i+1, j  );
                n8 = Im(i+1, j+1);
                n0 = Im(i,j);
                if n0 > n1 || n0 > n2 || n0 > n3 || n0 > n4 || n0 > n5 || n0 > n6 || n0 > n7 || n0 > n8
                    if n0 >= startline_thresh
                        i_temp = i;
                        j_temp = j;
                        count = 0;

                        while n0 > stopline_thresh && i_temp < m-1 && j_temp < n-1 && mark(i_temp, j_temp) ~= 1
                            count = count + 1;

                            mark(i_temp, j_temp) = 1;
                            edge(i_temp, j_temp) = 255;
                            ang = theta(i_temp,j_temp);

                            % next pixel
                            if((ang < -(180-22.5)) || (ang > (180-22.5)))
                                id = 1;
                                jd = -1;
                            else if(ang < -(90+22.5))
                                    id  = 1;
                                    jd = 0;
                                else if(ang < -(90-22.5))
                                        id = 1;
                                        jd = 1;
                                    else if(ang < -22.5)
                                            id = 0;
                                            jd = 1;
                                        else if(ang > 90+22.5)
                                                id = 0;
                                                jd = -1;
                                            else if(ang > 90-22.5)
                                                    id = -1;
                                                    jd = -1;
                                                else if(ang > 22.5)
                                                        id = -1;
                                                        jd = 0;
                                                    else
                                                        id = -1;
                                                        jd = 1;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            % move to that pixel
                            i_temp = i_temp + id;
                            j_temp = j_temp + jd;
                            % magnitude of the next pixel gradient
                            if(i_temp ~= 1 && j_temp ~= 1)
                                n0 = Im(i_temp, j_temp);
                                n1 = Im(i_temp-1, j_temp-1);
                                n2 = Im(i_temp-1, j_temp  );
                                n3 = Im(i_temp-1, j_temp+1);
                                n4 = Im(i_temp  , j_temp-1);
                                n5 = Im(i_temp  , j_temp+1);
                                n6 = Im(i_temp+1, j_temp-1);
                                n7 = Im(i_temp+1, j_temp  );
                                n8 = Im(i_temp+1, j_temp+1);
                            else
                                break;
                            end
                        end
                            % Remove stray dots
                            if(count <= 1)
                                edge(i,j) = 0;
                            end
                    end
                else
                    mark(i, j) = 1;
                end
            end
        end
    end
    
    
    figure();
    imshow(edge);
    saveas(imshow(edge), strcat(strrep(file.name, '.jpg',''), '_sol.jpg'));
end


