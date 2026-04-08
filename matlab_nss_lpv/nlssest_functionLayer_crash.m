%% Minimal repro: nlssest crashes with functionLayer + skip connections
%
% A dlnetwork with functionLayer and skip connections works fine with
% predict() and forward(), and assigns to idNeuralStateSpace.StateNetwork
% without error. But nlssest() crashes MATLAB with:
%
%   "Terminating because MCOS is being accessed on a non-MVM thread."
%
% The crash occurs because nlssest's internal training evaluates the
% functionLayer on a worker thread, but anonymous functions in
% functionLayer cannot run on non-main threads.
%
% MATLAB R2026a Prerelease Update 3 (26.1.0.3133997), macOS 26.4, ARM64
%
% Use case: LPV Neural ODE where MLP outputs gain-scheduled matrices
% and a final layer computes dx/dt = A(x,u)*x + B(x,u)*u + c(x,u)
% using skip connections from the original x and u inputs.

%% 1. Build network with functionLayer + skip connections
lgraph = layerGraph();
lgraph = addLayers(lgraph, featureInputLayer(2, 'Name', 'x'));
lgraph = addLayers(lgraph, featureInputLayer(1, 'Name', 'u'));
lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', 'concat'));
lgraph = addLayers(lgraph, fullyConnectedLayer(64, 'Name', 'fc1'));
lgraph = addLayers(lgraph, tanhLayer('Name', 'tanh1'));
lgraph = addLayers(lgraph, fullyConnectedLayer(8, 'Name', 'fc_raw'));

% This functionLayer uses skip connections from 'x' and 'u' inputs
lgraph = addLayers(lgraph, functionLayer(@(raw, xn, un) ...
    [raw(1,:).*xn(1,:) + raw(2,:).*xn(2,:) + raw(5,:).*un + raw(7,:); ...
     raw(3,:).*xn(1,:) + raw(4,:).*xn(2,:) + raw(6,:).*un + raw(8,:)], ...
    'Name', 'dxdt', 'NumInputs', 3, 'Formattable', true));

lgraph = connectLayers(lgraph, 'x', 'concat/in1');
lgraph = connectLayers(lgraph, 'u', 'concat/in2');
lgraph = connectLayers(lgraph, 'concat', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'tanh1');
lgraph = connectLayers(lgraph, 'tanh1', 'fc_raw');
lgraph = connectLayers(lgraph, 'fc_raw', 'dxdt/in1');
lgraph = connectLayers(lgraph, 'x', 'dxdt/in2');       % skip connection
lgraph = connectLayers(lgraph, 'u', 'dxdt/in3');       % skip connection

net = dlnetwork(lgraph);

%% 2. Verify network works standalone
y = predict(net, dlarray(randn(2,1,'single'),'CB'), dlarray(randn(1,1,'single'),'CB'));
fprintf('predict() works: output size = %s\n', mat2str(size(extractdata(y))));

%% 3. Assign to idNeuralStateSpace — works fine
nss = idNeuralStateSpace(2, NumInputs=1, NumOutputs=2, Ts=0);
nss.StateNetwork = net;
fprintf('StateNetwork assignment: OK\n');

%% 4. Create dummy training data
t = (0:0.001:0.1)';
u = 0.5 * ones(size(t));
y = [10*ones(size(t)), 2*ones(size(t))];
data = iddata(y, u, 0.001);

%% 5. Train — THIS CRASHES MATLAB
fprintf('Calling nlssest... (expect crash)\n');
opts = nssTrainingOptions('adam');
opts.MaxEpochs = 2;
nss = nlssest(data, nss, opts);  % <-- CRASH: MCOS on non-MVM thread
