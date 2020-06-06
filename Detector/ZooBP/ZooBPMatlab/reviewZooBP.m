function [userBelief,prodBelief,stats] = reviewZooBP( L,ep,userPrior,prodPrior,...
   maxIter)
%% reviewZooBP: ZooBP for 2 node and 2 edges types, bipartite network.
% L = adjacency list; Each edge is a tuple <userId, productId, 1/2>
% 1 : positive rating, 2 : negative
% ep : interaction strength, default value = 10^-4
% userPrior,userBelief : nUserx2 matrix of centered user priors (input),
% centered user final beliefs (output)
% prodPrior,prodBelief : nProdx2 matrix of centered product priors (input),
% centered product final beliefs (output)

if nargin < 5,
    maxIter = 50;
end
if nargin < 4,
    nProd = max(L(:,2)); prodPrior = zeros(nProd,2);
else
    nProd = size(prodPrior,1);
end
if nargin < 3,
    nUser = max(L(:,1)); userPrior = zeros(nUser,2);
else
    nUser = size(userPrior,1);
end
if nargin < 2,
    ep = 10^-2;
end

rating = L(:, 3);
L(rating == 1, 3) = -1;
L(rating == 2, 3) = -1;
L(rating == 3, 3) = -1;
L(L(:, 3) == -1, 3) = 2;
L(rating == 4, 3) = -2;
L(rating == 5, 3) = -2;
L(L(:, 3) == -2, 3) = 1;

[~,userID] = ismember(L(:,1),unique(L(:,1)));
[~,prodID] = ismember(L(:,2),unique(L(:,2)));
L(:, 1) = userID;
L(:, 2) = prodID;


rating = L(:,3);
Lpos = L(rating==1,1:2); Lneg = L(rating==2,1:2);
Apos = sparse(Lpos(:,1),Lpos(:,2),1,nUser,nProd);
Aneg = sparse(Lneg(:,1),Lneg(:,2),1,nUser,nProd);
disp(Lneg(1:5, :));
clearvars L Lpos Lneg rating
%% Prior Beliefs
E = [reshape(userPrior',[],1);reshape(prodPrior',[],1)];
clearvars userPrior prodPrior
%% P
H = [0.5,-0.5;-0.5,0.5]; R = kron(Apos-Aneg, ep*H);
P = [sparse(nUser*2,nUser*2) 0.5 * R;
    0.5*transpose(R) sparse(nProd*2,nProd*2)];
clearvars R
%% Q
D12 = diag(sparse(sum(Apos+Aneg,2))); D21 = diag(sparse(sum(Apos+Aneg,1)));
Q_1 = speye(nUser*2) + 0.25 * ep^2 * kron(D12,H);
Q_2 = speye(nProd*2) + 0.25 * ep^2 * kron(D21,H);
clearvars D12 D21
Q = sparse([Q_1     sparse(nUser*2,nProd*2);
           sparse(nProd*2,nUser*2)      Q_2]);
clearvars Q_1 Q_2
%% Iterative Solution
M = P - Q + speye((nUser+nProd)*2); 
disp(M(1:5));
clearvars P Q
B = initializeFinalBeliefs(nUser,nProd,10^-3);
res = 1; iter=0;
vals = eigs(M); stats.ei = vals(1);
tic;
while res>10^-8 %|| iter < 3
     Bold = B;
     B = E + M*Bold;
     res = sum(sum(abs(Bold-B)));
     iter = iter+1;
     if iter == maxIter
         break;
     end
end
stats.elapsedTime = toc; stats.nIter = iter;
%% Final Beliefs
B1 = B(1:nUser*2); B2 = B(nUser*2+1:end);
userBelief = reshape(B1,2,nUser)'; prodBelief = reshape(B2,2,nProd)';
end

function B = initializeFinalBeliefs(N1,N2,m)
    r1 = m*(rand(N1,1)-0.5); r2 = m*(rand(N2,1)-0.5);
    B1 = [r1,-r1]; B2 = [r2,-r2];
    B = [reshape(B1',[],1);reshape(B2',[],1)]; 
end