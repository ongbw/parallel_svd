function [] = run03(M,d,D,N)
% function [] = run03(M,d,D,N)
% parallel one-level SVD formulation
% M: number of blocks (h5 files created)
% d: instrinsic dimensionality
% D: ambient dimensionality
% N: number of samples

ntrials = 1; % for timing results

WRITE_HDF5 = 0;

% check what inputs the user passed in
switch nargin
case 0
  % set these default values
  M = 10;
  d = 20;
  D = 100;
  N = 200;


case 1
  % set these default values
  d = 20;
  D = 100;
  N = 200;

case 2
  D = 10*d;
  N = 200;

case 3
  N = 200;
end % switch

% Create  d-dim subspace
basis=randn(D,d);

for m = 1:M
    % Create coordinates for d-dim dataset X
    coords{m}=rand(d,N);
    % Create d-dim dataset X
    Ai{m}=basis*coords{m};

    if WRITE_HDF5
      % write data set to an hdf5 file
      filename = ['matrix_',num2str(m),'.h5'];
      h5create(filename,'/TestSet',[D N]);
      h5write(filename,'/TestSet',Ai{m});
      h5disp(filename);
   end
end


A = zeros(D,N*M);
for m = 1:M
    A(:,(m-1)*N+[1:N]) = Ai{m};
end

if WRITE_HDF5
  % store full matrix to hdf file
  filename = 'full.h5';
  h5create(filename,'/TestSet',[D N*M]);
  h5write(filename,'/TestSet',A);
  h5disp(filename);
end

tic
%Find SVD of [Ai{1} Ai{2} ... Ai{M}]
for k = 1:ntrials
    [UC,SC,VC] = svd([A]);
end
s = toc;
fprintf('time to find full SVDs (%d trials): %f seconds \n',ntrials,s);


tic
for k = 1:ntrials
    % Find SVD of distributed dataset
    for m = 1:M
        [U{m},S{m},V{m}] = svd(Ai{m});        
        %svd(Ai{m});
    end
end
s = toc;
fprintf('time to find distributed SVDs (%d trials): %f seconds \n',ntrials,s);


% Specify proxy data set
Proxy = zeros(D,d*M);
for m = 1:M
    Proxy(:,(m-1)*d+[1:d]) = U{m}(:,1:d)*S{m}(1:d,1:d);
end


tic
% find svd of proxy data set
for k = 1:10
    [UP,SP,VP] = svd(Proxy);
end
s = toc;
fprintf('time to find SVDs of proxy matrices (%d trials): %f seconds \n',ntrials,s);

disp('difference in singular values')
norm(SC(:,1:d)-SP(:,1:d))

disp('difference in left singular vectors')
scaling = diag(sign(UP(1,1:d)).*sign(UC(1,1:d)));
norm(UP(:,1:d)-UC(:,1:d)*scaling)

% if right singular vectors are desired ...
% compute right singular vectors
RV = zeros(size(VC(:,1:d)));
for k = 1:d
    RV(:,k) = A'*UC(:,k)/SC(k,k);
end

    
disp('difference in right singular vectors')
norm(RV(:,1:d)-VC(:,1:d))


