function [R_d_min,R_d,S_d_min]=CF_downlink_ZF(M,K,tau_cf,Ps,beta,g)
% cell-free massive MIMO system with zero forcing beamforming
% output incloud worst case rate, throughput and rate for all users
%% parameters
B=2e+7;

tau_c=200;

% normalized SNRs
k_B=1.381e-23;
T_0=290;
NF=10^(9/10); 
NP=B*k_B*T_0*NF;
rho_d=Ps/NP/1000;
rho_p=Ps/NP/1000;
%% uplink training

%pilot assignment initialization phi_k
[s_phi,~,~]=svd(randn(tau_cf,tau_cf));
phi=zeros(tau_cf,K);
for k=1:K
    phi(:,k)=s_phi(:,randi([1,tau_cf]));
end
phi=phi(:,randperm(K));

z_p=1/sqrt(2).*randn(M,tau_cf)+1i*1/sqrt(2).*randn(M,tau_cf);

% channel estimation
y_p=sqrt(tau_cf*rho_p)*g*phi'+z_p;
y_p_mk=y_p*phi;

c=zeros(M,K);
for m=1:M
    for k=1:K
        c(m,k)=sqrt(tau_cf*rho_p)*beta(m,k)/(tau_cf*rho_p*beta(m,:)*(abs(phi(:,k)'*phi).^2)'+1);
    end
end
% g_est=c.*y_p_mk;
alpha=sqrt(tau_cf*rho_p).*beta.*c;

%% take approximation

gamma = zeros(K,K);

for k = 1:K
    var_e = diag(beta(:,k)-alpha(:,k));
    temp_gamma = zeros(K,K);
    for iter =1:50
        g_dummy = sqrt(alpha/2).*(randn(M,K)+1i*randn(M,K));
        g_inv = inv(real(g_dummy.'*conj(g_dummy)));
        temp_gamma = temp_gamma+real(g_inv*g_dummy.'*var_e*conj(g_dummy)*g_inv);
    end
    gamma(:,k) = diag(temp_gamma/50);
end

delta = zeros(M,K);
for m = 1:M
    temp_delta = zeros(K,K);
    for iter =1:50
        g_dummy = sqrt(alpha/2).*(randn(M,K)+1i*randn(M,K));
        g_inv = inv(real(g_dummy.'*conj(g_dummy)));
        temp_delta = temp_delta+real(g_inv*(g_dummy(m,:)'*g_dummy(m,:))*g_inv);
    end
    delta(m,:) = diag(temp_delta/50)';
end

eta = 1/max(sum(delta,2))*ones(K,1);
%% downlink optimization
t_min=0; t_max=2.^(4*log10(M)-1);
epsilon=t_max*(2^-10);

while (t_max-t_min)>=epsilon
    t_cvx=(t_min+t_max)/2;
    
    cvx_begin quiet
    variable eta_cvx(K,1)

    maximize 1;
    subject to
    
    for m=1:M
        0 <= delta(m,:)*eta_cvx <= rho_d;
    end
    
    for k=1:K
        ds=eta_cvx(k);
        n1=sum(eta_cvx.*gamma(:,k));
        ds/t_cvx-(1+n1)>=0;
    end 
    
    cvx_end
    
    eta_iter=eta_cvx;
    if strcmp(cvx_status,'Solved')
        eta=eta_iter;
        t_min=t_cvx;   
    else t_max=t_cvx;
    end
end
%% downlink rate

R_d_min = log2(1+t_cvx);
S_d_min=B/1000000*(1-tau_cf/tau_c)/2*R_d_min;
SINR = zeros(K,1);
for k=1:K
    ds = rho_d*eta(k);
    n1 = rho_d*sum(eta.*gamma(:,k));
    SINR(k)=ds/(n1+1);
end
R_d=log2(1+SINR);

