seed = 10;
rng(seed);

%% Constants and connectivity 
tau  = 0.01;
N_G  = 1000;
p_GG = 0.1;
p_z  = 1;
g_Gz = 2; % If zero, there is no feedback.
g_GF = 0;
g_GG = 1.5;
a    = 1.0;

J_GG = randn(N_G).*(rand(N_G)<=p_GG)/sqrt(p_GG*N_G);
J_Gz = 2*rand(N_G,1)-1;

%% Time-related variables
T_end = 20.0;
dt    = tau/4;
t     = 0:dt:T_end;
num_iters = length(t);

%% The target function: a random periodic function with period T_end/2
num_comps = 2; % Number of frequency components included

f = 0*t;
f0 = 1/(T_end/4); % Fundamental frequency
for i = 1:num_comps
    fi = randi(4*num_comps)*f0; % pick a random harmonic
    ai = rand(); % random amplitude
    phi = rand*2*pi; % random phase
    f = f + ai*cos(2*pi*fi*t + phi); % add the component
end

%% Run the network
x_hist = randn(N_G,num_iters); 
z_hist = zeros(1,  num_iters);
em_hist = zeros(1,num_iters);
ep_hist = zeros(1,num_iters);
w_hist = 0*x_hist;
dw_hist = w_hist;

x = randn(N_G,1);
w = randn(N_G,1)/sqrt(N_G);
P = eye(N_G)/a;

learn = 1; % Set to non-zero to learn w.

fprintf('Running FORCE.\n');
for i = 1:num_iters
    if (mod(i,100)==0)
        fprintf('.');
    end

    r = tanh(x);
    z = w'*r;
    x_hist(:,i) = x;
    z_hist(i) = z;
    w_hist(:,i) = w;
    dxdt = -x + g_GG*J_GG*r + g_Gz*J_Gz*z;
    dxdt = dxdt/tau;
    x = x + dxdt*dt;
    
    if i<num_iters/2
        e_minus = z - f(i);
        em_hist(i) = e_minus;
        Pr = P*r;
        dP = -Pr*Pr'/(1 + r'*Pr);
        P = P + dP;
        dw = -e_minus*P*r;
        dw_hist(:,i) = dw;
        w = w + dw;
        ep_hist(i) = w'*r - f(i);
    end
    
    
end
r_hist = tanh(x_hist);
fprintf('\nDone.\n');
fprintf('MSE between target and output: %1.3e\n', mean((f - z_hist).^2))
%% Plot the results
clf;
subplot(2,1,1);
plot(t,f, 'LineWidth',2);
hold on;
plot(t,z_hist,'LineWidth',2);
legend('target','output')
subplot(2,1,2);
plot(sqrt(sum(dw_hist.^2,1)))