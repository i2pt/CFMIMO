function [R_d_min,R_d,S_d_min]=CF_downlink_CB(M,K,tau_cf,Ps,beta,g)
% cell-free massive MIMO system with conjugate beamforming
% output incloud worst case rate, throughput and rate for all users
%% parameters
% Fixed Parameters
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

% initial lower bound, uniform power
eta_up=1./sum(gamma,2)*ones(1,K);
SINR=zeros(1,K);
for k=1:K
    I_ki=ones(1,K);
    I_ki(k)=0;
    n_1=rho_d*(I_ki.*(beta(:,k)'*(eta_up.^0.5.*gamma./beta)).^2)*(abs(phi'*phi(:,k)).^2);
    n_2=rho_d*sum(beta(:,k)'*(eta_up.*gamma));
    s=rho_d*((eta_up(:,k)'.^0.5)*gamma(:,k))^2;
    SINR(k)=s/(n_1+n_2+1);
end
t_min=0;
t_max=2^(4*min(SINR));
epsilon=t_max*2^-10;

gamma_s=rho_d*gamma;
beta_s=rho_d*beta;

while (t_max-t_min)>=epsilon
    t_cvx=(t_min+t_max)/2;
    
    cvx_begin quiet
    variable v(M+K,K+1);
    % v 1:M,1:K zta, 1:M,K+1 nu
    % v M+1:M+K,1:K sigma
    maximize sum(v(1:M,K+1));
    subject to
    
    sum(gamma_s.*(v(1:M,1:K).^2),2)-v(1:M,K+1)<=0; % constraint 2
    0<=v(1:M,K+1)<=1; % constraint 4
    v(1:M,1:K)>=0; % constraint 5
    
    I_kk=eye(K);
    for k=1:K
        I_k=I_kk(:,[1:k-1 k+1:K]);
        v_k1=(phi'*phi(:,k).*v(M+1:M+K,k)).';
        v_k2=(sqrt(beta_s(:,k)).*v(1:M,K+1)).';
        v_k=[v_k1*I_k v_k2 1];
        norm(v_k)-gamma_s(:,k)'*v(1:M,k)/sqrt(t_cvx)<=0; % constraint 1
        for kk=1:K
            if kk~=k
                sum(gamma_s(:,kk).*v(1:M,kk)./beta_s(:,kk).*beta_s(:,k))-v(M+kk,k)<=0; % constraint 3
            end
        end
    end 
    
    cvx_end

    if 1e+3*mean(sum((gamma_s.*(v(1:M,1:K).^2)),1))>1
        if sum(gamma_s.*(v(1:M,1:K).^2),2)<=v(1:M,K+1).^2
            eta_d=(v(1:M,1:K).^2)*rho_d;
            t_min=t_cvx;
        else t_max=t_cvx;
        end
    else t_max=t_cvx;
    end
end
%% downlink rate

R_d_min=log2(1+t_cvx);
S_d_min=B/1000000*(1-tau_cf/tau_c)/2*R_d_min;
for k=1:K
    I_ki=ones(1,K);
    I_ki(k)=0;
    n_1=rho_d*(I_ki.*(beta(:,k)'*(eta_d.^0.5.*gamma./beta)).^2)*(abs(phi'*phi(:,k)).^2);
    n_2=rho_d*sum(beta(:,k)'*(eta_d.*gamma));
    s=rho_d*((eta_d(:,k)'.^0.5)*gamma(:,k))^2;
    SINR(k)=s/(n_1+n_2+1);
end
R_d = log2(1+SINR);
