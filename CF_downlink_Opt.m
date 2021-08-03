function [R_d_min,R_d,S_d_min]=CF_downlink_Opt(M,K,tau_cf,Ps,beta,g)
% cell-free massive MIMO system with optimum beamforming
% output incloud worst case rate, throughput and rate for all users
%% parameters
% Fixed Parameters
%f=1.9e+3;
B=2e+7; % Bandwith 20MHz

tau_c=200; % tau_c 200 coherence

% normalized SNRs
k_B=1.381e-23;
T_0=290;
NF=10^(9/10); % noise figure 9 dB
NP=B*k_B*T_0*NF;
rho_d=Ps/NP/1000;
rho_p=Ps/NP/1000;
%% uplink training

%pilot assignment initialization phi_k
[s_phi,~,~]=svd(randn(tau_cf,tau_cf)); % orthogonal pilot sequences set
phi=zeros(tau_cf,K);
for k=1:K
    phi(:,k)=s_phi(:,randi([1,tau_cf]));
end
phi=phi(:,randperm(K));% 20*40 pilots 

z_p=1/sqrt(2).*randn(M,tau_cf)+1i*1/sqrt(2).*randn(M,tau_cf); % tao_cf length cn noise for m APs

% channel estimation
y_p=sqrt(tau_cf*rho_p)*g*phi'+z_p; % 100*20 received pilot at m APs
y_p_mk=y_p*phi; % M*K projection of received pilot

c=zeros(M,K); % LMMSE
for m=1:M
    for k=1:K
        c(m,k)=sqrt(tau_cf*rho_p)*beta(m,k)/(tau_cf*rho_p*beta(m,:)*(abs(phi(:,k)'*phi).^2)'+1);
    end
end
g_est=c.*y_p_mk; % estimated channel coefficient
gamma=sqrt(tau_cf*rho_p).*beta.*c;
%% downlink optimization

t_min=0; t_max=2.^(4*log10(M)-1);
epsilon=t_max*(2^-10);

% cvx_solver SDPT3
while (t_max-t_min)>=epsilon
    t_cvx=(t_min+t_max)/2;
    
    cvx_begin quiet
    variable w_cvx(M,K) complex
    
    maximize 1;
    subject to

    for m=1:M
        w_cvx(m,:)*w_cvx(m,:)'<=1; % const 3
    end
    
    for k=1:K
        I_k=eye(K);
        I_k(:,k)=[];
        mui=g_est(:,k).'*(w_cvx*I_k);
        v2=reshape(mui,(K-1),1);
        
        cee=(sqrt(beta(:,k)-gamma(:,k))*ones(1,K)).*w_cvx;
        v1 = reshape(cee,M*K,1);
        
        v_k=[sqrt(rho_d)*v1' sqrt(rho_d)*v2' 1]';
        norm(v_k)-1/sqrt(t_cvx)*sqrt(rho_d)*real(sum(g_est(:,k).*w_cvx(:,k)))<=0; % const 1
    end 
    
    cvx_end
    
    w_iter=w_cvx;
    if trace(w_iter*w_iter')>1
        w=w_iter;
        t_min=t_cvx;
    else t_max=t_cvx;
    end
end

%% downlink rate

R_d_min=log2(1+t_cvx);
S_d_min=B/1000000*(1-tau_cf/tau_c)/2*R_d_min;

SINR=zeros(K,1);
for k=1:K
    ds=norm(sum(g_est(:,k).*w(:,k)))^2;
    cee = (sqrt(beta(:,k)-gamma(:,k))*ones(1,K)).*w;
    n1 = norm(reshape(cee,M*K,1))^2;
    I_k=eye(K);
    I_k(:,k)=[];
    mui = (g_est(:,k).')*(w*I_k);
    n2 = norm(mui')^2;

    SINR(k)=ds/(n1+n2+1/rho_d);
end                 
R_d=log2(1+SINR);
