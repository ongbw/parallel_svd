function [] = run02(M)
% run02.m
% simple numerical test for "merging" M d-dimensional subspaces

% set default value for M if M not specified.
if nargin < 1
  M =2;
end

d = 3;
D = 24;
N = 3;

for m = 1:M
    % Create  d-dim subspace
    basis{m}=randn(D,d);
    % Create coordinates for d-dim dataset X
    coords{m}=rand(d,N);
    % Create d-dim dataset X
    Ai{m}=basis{m}*coords{m};
end

A = zeros(D,N*M);
for m = 1:M
    A(:,(m-1)*N+[1:N]) = Ai{m};
end

% Find SVD of [Ai{1} Ai{2} ... Ai{M}]
[UC,SC,VC] = svd([A]);

% Find SVD of distributed dataset
for m = 1:M
    [U{m},S{m},V{m}] = svd(Ai{m});
end

% Find proxy data set
Proxy = zeros(D,d*M);
for m = 1:M
    Proxy(:,(m-1)*d+[1:d]) = U{m}(:,1:d)*S{m}(1:d,1:d);
end

% find svd of proxy data set
[UP,SP,VP] = svd(Proxy);

disp('difference in singular values')
norm(SC(:,1:d)-SP(:,1:d))

% note, if there are repeated singular values, then 
% singular vectors are similar up to a unitary transformation
% disp('difference in singular vectors')
% [UP(:,1:d),zeros(D,2),UC(:,1:d)]
