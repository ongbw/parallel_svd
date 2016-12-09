function [] = run01()
% run01.m
% simple numerical test for "merging" two d-dimensional subspaces

d = 3;
D = 24;
N = 100;

% Create Basis for first d-dim subspace
Xbasis=rand(D,d);
% Create coordinates for d-dim dataset X
Xcoords=rand(d,N);
% Create d-dim dataset X
X=Xbasis*Xcoords;

% Create Basis for second d-dim subspace
Ybasis=rand(D,d);
% Create coordinates for d-dim dataset Y
Ycoords=rand(d,N);
% Create d-dim dataset Y
Y=Ybasis*Ycoords;

% Find SVD of [X Y]
[UC,SC,VC] = svd([X,Y]);

% Form the proxy dataset
[UX,SX,VX] = svd(X);
[UY,SY,VY] = svd(Y);
Proxy=[(UX(:,1:d)*SX(1:d,1:d)),(UY(:,1:d)*SY(1:d,1:d))];
[UP,SP,VP] = svd(Proxy);

% calculate norm of the difference of singular values
norm(SC(:,1:2*d)-SP)
