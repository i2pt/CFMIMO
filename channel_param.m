function [beta,g] = channel_param(M,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random channel parameters genrated based on system size %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = 1000; % 1 km square area

loca_AP=unifrnd(0,D,M,2);
loca_AP=[loca_AP,15*ones(M,1)]; % location of APs

loca_u=unifrnd(0,D,K,2);
loca_u=[loca_u,1.65*ones(K,1)]; % location of users

% wrapped around distance
loca_AP = repmat(reshape(loca_AP,M,1,3),[1 K 1]);
loca_u = repmat(reshape(loca_u,1,K,3),[M 1 1]);
loca_dif = abs(loca_AP-loca_u);
loca_dif_wrap = loca_dif.*(loca_dif<=(D/2))+(repmat(cat(3,D,D,0),[M K 1])-loca_dif).*(loca_dif>(D/2));
% distance of AP to user in m 
d = sqrt(loca_dif_wrap(:,:,1).^2+loca_dif_wrap(:,:,2).^2+loca_dif_wrap(:,:,3).^2); 
d = d*1e-3; % distance in km

L = 140.72;
PL_dB=-L-35*log10(d); % path loss in dB
z = 8*randn(M,K); % shadowing in dB
beta = 10.^((PL_dB+z)/10); % large small scale fading

h = 1/sqrt(2).*randn(M,K)+1i*1/sqrt(2).*randn(M,K); % small scale fading

g = sqrt(beta).*h; % channel coefficient

