function [points rec_err] = find_3d_points(P1, P2, matches)
%FIND_3D_POINTS Summary of this function goes here
%   Detailed explanation goes here
    [N, d] = size(matches);

    points = zeros(N, 3);

    % loop over all points in matches
    for i = 1:N
        x1 = matches(i, 1);
        y1 = matches(i, 2);
        x2 = matches(i, 3);
        y2 = matches(i, 4);

        A = [(x1*P1(3,1)-P1(1,1)) (x1*P1(3,2)-P1(1,2)) (x1*P1(3,3)-P1(1,3));
             (y1*P1(3,1)-P1(2,1)) (y1*P1(3,2)-P1(2,2)) (y1*P1(3,3)-P1(2,3));
             (x2*P2(3,1)-P2(1,1)) (x2*P2(3,2)-P2(1,2)) (x2*P2(3,3)-P2(1,3));
             (y2*P2(3,1)-P2(2,1)) (y2*P2(3,2)-P2(2,2)) (y2*P2(3,3)-P2(2,3));    
            ];

        b = -[(x1*P1(3,4)-P1(1,4));
              (y1*P1(3,4)-P1(2,4));
              (x2*P2(3,4)-P2(1,4));
              (y2*P2(3,4)-P2(2,4))
            ];

        [U S V] = svd(A);

        S_aug = [S((1:3),:)^-1 zeros(3,1)];

        points(i,:) = V*S_aug*U'*b;
    end
    
    % Reconstruction error:
    X1 = (P1*[points ones(N, 1)]')';
    X2 = (P2*[points ones(N, 1)]')';
    
    
    rec_err = sum(sqrt(sum((X1(:,[1,2])-matches(:,[1,2])).^2)) + sqrt(sum((X2(:,[1,2])-matches(:,[3,4])).^2)))/(2*N);
end

