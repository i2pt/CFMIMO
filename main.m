clc;
clear all;

%% Parameters

M=100; % number of APs
K=20; % number of UEs
tau_cf=[K/2 K]; % training pilot length
Ps=23; % dBm
Ps=10.^(Ps./10);
maxiter = 200; % simulation run for 200 realizations

% downlink throughput CDF with power control
S_min_OB=zeros(length(tau_cf),maxiter);
S_min_CB=zeros(length(tau_cf),maxiter);
S_min_ZF=zeros(length(tau_cf),maxiter);

parfor iter=1:maxiter
    iter
    
    S_temp_OB=zeros(length(tau_cf),1);
    for i=1:length(tau_cf)
        [beta,g] = channel_param(M,K);
        [~,~,S_temp_OB(i)]=CF_downlink_Opt(M,K,tau_cf(i),Ps,beta,g);
    end
    S_min_OB(:,iter)=S_temp_OB;
    
    S_temp_CB=zeros(length(tau_cf),1);
    for i=1:length(tau_cf)
        [beta,g] = channel_param(M,K);
        [~,~,S_temp_CB(i)]=CF_downlink_CB(M,K,tau_cf(i),Ps,beta,g);
    end
    S_min_CB(:,iter)=S_temp_CB;
    
    S_temp_ZF=zeros(length(tau_cf),1);
    for i=1:length(tau_cf)
        [beta,g] = channel_param(M,K);
        [~,~,S_temp_ZF(i)]=CF_downlink_ZF(M,K,tau_cf(i),Ps,beta,g);
    end
    S_min_ZF(:,iter)=S_temp_ZF;
end

figure()
hold on
cdfplot(S_min_OB(1,:));
cdfplot(S_min_OB(2,:));
cdfplot(S_min_CB(1,:));
cdfplot(S_min_CB(2,:));
cdfplot(S_min_ZF(1,:));
cdfplot(S_min_ZF(2,:));
xlabel('Per-user throughput')
ylabel('CDF')
legend('OB, \tau=K/2','OB, \tau=K','CB, \tau=K/2','CB, \tau=K','ZF, \tau=K/2','ZF, \tau=K')

