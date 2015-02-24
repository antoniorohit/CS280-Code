% TOTEST
axis_vector=zeros(1,3);
axis_vector(1)=1;
theta=1.7;

% initialize result
R=zeros(3);

% create new vector with the rotation angle as its norm
vec = theta * axis_vector;

% first find the cross_product matrix associated with the cross-product
% with the vector just created

cross_prod_mat = zeros(3);
cross_prod_mat(1,2)=-vec(3);
cross_prod_mat(1,3)=vec(2);
cross_prod_mat(2,1)=vec(3);
cross_prod_mat(2,3)=-vec(1);
cross_prod_mat(3,1)=-vec(2);
cross_prod_mat(3,2)=vec(1);

cross_prod_mat

% now compute the exponential of this matrix
R=expm(cross_prod_mat)

% find eigen values
[V,D]=eig(R)

% retrieve angle and compare it to original one
cos_phi = 0.5 * (trace(R) - 1);
cos(theta);
abs(theta-cos_phi)

% retrieve both angle and axis vector
diff_mat = R - R';
two_sin_phi=norm(diff_mat);
angle=asin(0.5*two_sin_phi);
skew_mat=diff_mat ./ two_sin_phi;
new_axis=zeros(1,3);
new_axis(1)=skew_mat(3,2);
new_axis(2)=skew_mat(1,3);
new_axis(3)=skew_mat(2,1);

new_axis


