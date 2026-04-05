function train_lpv_rk4(seed, opts)
% train_lpv_rk4  LPV Neural ODE training in MATLAB DLT — replicates PyTorch exactly.
%
% Architecture: dx/dt = dxdt_scale .* (A(x,u)*x_n + B(x,u)*u_n + c(x,u))
%   MLP: [x_n, u_n] -> [A(4), B(2), c(2)] = 8 outputs
%
% Replicates pytorch_neural_ode_v2/train_neuralode_lpv.py feature-for-feature.
% Same as train_mlp_rk4.m but with LPV architecture and initialization.
%
% Usage:
%   train_lpv_rk4(7)                     % seed 7 (PyTorch best LPV)
%   train_lpv_rk4(7, 'maxHours', 10)
%   train_lpv_rk4(0, 'test', true)

arguments
    seed (1,1) double = 7
    opts.maxHours (1,1) double = 10
    opts.test (1,1) logical = false
end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

TS=5e-6; SKIP=10; DT=TS*SKIP; NX=2; H=64; GRAD_CLIP=1.0;
VAL_FREQ=10; EARLY_STOP=20; CP_MIN=15;
VAL_IDX = [5, 12, 14];

CP_DIR = fullfile(thisDir, 'checkpoints_lpv', sprintf('run_%04d', seed));
if ~isfolder(CP_DIR), mkdir(CP_DIR); end
maxNumCompThreads(1); rng(seed);

fprintf('=== LPV RK4 Training (DLT V2) — Seed %d ===\n', seed);

%% ── Load data & compute norm stats from train only ─────────────────────────
csvDir = fullfile(repoRoot, 'data', 'neural_ode');
csvFiles = dir(fullfile(csvDir, 'profile_*.csv'));
[~,ord] = sort({csvFiles.name}); csvFiles = csvFiles(ord);
nTotal = numel(csvFiles);
profiles_u = cell(nTotal,1); profiles_y = cell(nTotal,1);
for k=1:nTotal
    raw = single(readmatrix(fullfile(csvDir, csvFiles(k).name)));
    profiles_u{k}=raw(:,1); profiles_y{k}=raw(:,2:3);
end

TRAIN_IDX = setdiff(1:nTotal, VAL_IDX); nTrain = numel(TRAIN_IDX);
all_u=[]; all_y=[]; all_dv=[]; all_di=[];
for i=TRAIN_IDX
    all_u=[all_u;profiles_u{i}]; all_y=[all_y;profiles_y{i}];
    all_dv=[all_dv;diff(profiles_y{i}(:,1))/TS]; all_di=[all_di;diff(profiles_y{i}(:,2))/TS];
end
ns.u_mean=single(mean(all_u)); ns.u_std=single(std(all_u));
ns.x_mean=single(mean(all_y))'; ns.x_std=single(std(all_y))';
ns.dxdt_scale=single([std(all_dv);std(all_di)]);
save(fullfile(CP_DIR,'norm_stats.mat'),'-struct','ns');

train_u=profiles_u(TRAIN_IDX); train_y=profiles_y(TRAIN_IDX);
val_u=profiles_u(VAL_IDX); val_y=profiles_y(VAL_IDX);
fprintf('  Train: %d, Val: %d | dxdt_scale=[%.0f,%.0f]\n', nTrain, numel(VAL_IDX), ns.dxdt_scale);

jacFile = fullfile(repoRoot, 'data', 'neural_ode', 'jacobian_targets.json');
hasJac=false; A_tgt=[]; x_ops=[]; u_ops=[];
if isfile(jacFile)
    jd=jsondecode(fileread(jacFile));
    A_tgt=single(jd.A_targets); x_ops=single(jd.x_ops); u_ops=single(jd.u_ops(:));
    if ndims(A_tgt)==3&&size(A_tgt,1)==2, A_tgt=permute(A_tgt,[3 1 2]); end
    hasJac=true; fprintf('  Jacobian: %d points\n', size(A_tgt,1));
end

%% ── Build LPV dlnetwork (3->8 outputs) ─────────────────────────────────────
layers = [
    featureInputLayer(NX+1,'Name','input','Normalization','none')
    fullyConnectedLayer(H,'Name','fc1')
    tanhLayer('Name','tanh1')
    fullyConnectedLayer(H,'Name','fc2')
    tanhLayer('Name','tanh2')
    fullyConnectedLayer(NX*NX+NX+NX,'Name','fc3')  % 8 outputs: A(4)+B(2)+c(2)
];
net = dlnetwork(layers); net = initialize(net);

% LPV-specific initialization: 12 A-matrices × 4 neurons = 48 neurons
if hasJac
    nPts = size(A_tgt,1);
    neurPerPt = min(4, H/nPts);
    W1 = extractdata(net.Learnables.Value{1}); b1 = extractdata(net.Learnables.Value{2});
    W1(:,:)=0; b1(:)=0;
    for i=1:nPts
        xn = (x_ops(i,:)' - ns.x_mean) ./ ns.x_std;
        dn = (u_ops(i) - ns.u_mean) / ns.u_std;
        xu = [xn; dn];
        for j=1:neurPerPt
            row = (i-1)*neurPerPt + j;
            if row > H, break; end
            W1(row,:) = xu' * (0.5 + 0.5*(j-1));
            b1(row) = -dot(W1(row,:), xu) * (0.3 + 0.2*(j-1));
        end
    end
    filled = nPts*neurPerPt;
    if filled < H
        W1(filled+1:end,:) = 0.05*randn(H-filled, NX+1, 'single');
        b1(filled+1:end) = 0.05*randn(H-filled, 1, 'single');
    end
    net.Learnables.Value{1} = dlarray(W1);
    net.Learnables.Value{2} = dlarray(b1);
    fprintf('  LPV init: %dx%d=%d neurons from A-matrices\n', nPts, neurPerPt, min(nPts*neurPerPt,H));
end
fprintf('  Parameters: %d\n', sum(cellfun(@numel, {net.Learnables.Value{:}})));

%% ── Filters ────────────────────────────────────────────────────────────────
f_cut=sqrt(600*200e3); fs_eff=1/DT; fn=min(f_cut/(fs_eff/2),0.95);
if fn>0.05
    [b_lp,a_lp]=butter(2,fn,'low'); [b_hp,a_hp]=butter(2,fn,'high');
    b_lp=single(b_lp);a_lp=single(a_lp);b_hp=single(b_hp);a_hp=single(a_hp);
else, b_lp=[];a_lp=[];b_hp=[];a_hp=[]; end

%% ── Phases ─────────────────────────────────────────────────────────────────
if opts.test
    phases={struct('name','TEST','epochs',5,'window_ms',5,'lr',1e-3,'lambda_J',0,'lambda_ripple',0,'win_per_prof',1)};
    VAL_FREQ=1; EARLY_STOP=999;
else
    phases={
        struct('name','Phase1_HalfOsc','epochs',300,'window_ms',5,'lr',1e-3,'lambda_J',0.1,'lambda_ripple',0,'win_per_prof',3)
        struct('name','Phase2_FullOsc','epochs',500,'window_ms',20,'lr',5e-4,'lambda_J',0.01,'lambda_ripple',0.1,'win_per_prof',3)
        struct('name','Phase3_Finetune','epochs',200,'window_ms',20,'lr',1e-4,'lambda_J',0,'lambda_ripple',0.1,'win_per_prof',3)
    };
end

bestVal=inf; bestNet=net; trainHist=[]; valHist=[]; totalEpochs=0;
tStart=tic; tLastCp=tic;
accLoss = dlaccelerate(@computeLPVLossAndGrad);

%% ── Phases 1-3 ─────────────────────────────────────────────────────────────
for phIdx=1:numel(phases)
    ph=phases{phIdx}; winSamp=round(ph.window_ms*1e-3/TS);
    lr_max=ph.lr; lr_min=lr_max*0.01; nEp=ph.epochs;
    clearCache(accLoss);
    fprintf('\n--- %s: %d ep | win=%dms | LR=%.1e->%.1e ---\n', ph.name, nEp, ph.window_ms, lr_max, lr_min);

    avgG=[]; avgSqG=[]; phIter=0; noImp=0;
    for epoch=1:nEp
        totalEpochs=totalEpochs+1; tEp=tic;
        lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(epoch-1)/nEp));

        allWin = collectWindows(train_u, train_y, nTrain, winSamp, ph.win_per_prof);
        epochLoss=0; nWin=0;
        for w=1:size(allWin,1)
            u_w=allWin{w,1}; y_w=allWin{w,2};
            x0=dlarray(y_w(1,:)');
            phIter=phIter+1;
            [loss,grads] = dlfeval(accLoss, net, x0, dlarray(u_w), dlarray(y_w), ...
                ns, DT, SKIP, ph.lambda_ripple, b_lp, a_lp, b_hp, a_hp);
            if isnan(extractdata(loss))||isinf(extractdata(loss)), continue; end
            grads=clipGrads(grads,GRAD_CLIP);
            [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,phIter,lr);
            epochLoss=epochLoss+extractdata(loss); nWin=nWin+1;

            if hasJac && ph.lambda_J>0 && mod(nWin,5)==0
                phIter=phIter+1;
                [lJ,gJ]=dlfeval(@computeLPVJacLoss, net, A_tgt, x_ops, u_ops, ns, ph.lambda_J);
                if ~isnan(extractdata(lJ))
                    gJ=clipGrads(gJ,GRAD_CLIP);
                    [net,avgG,avgSqG]=adamupdate(net,gJ,avgG,avgSqG,phIter,lr);
                end
            end
        end
        trainHist(end+1)=epochLoss/max(nWin,1); epTime=toc(tEp); %#ok<AGROW>

        if mod(epoch,VAL_FREQ)==0||epoch==1||epoch==nEp
            [vl,rv,ri]=computeVal(net,val_u,val_y,ns,DT,SKIP);
            [fpv,fpi]=computeFP(net,val_u,val_y,ns,DT,SKIP);
            valHist(end+1)=vl; %#ok<AGROW>
            imp='';
            if vl<bestVal, bestVal=vl;bestNet=net;noImp=0;imp=' ** BEST **';
                saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
            else, noImp=noImp+1; end
            elapsed=toc(tStart)/3600;
            fprintf('  [%s] Ep %3d/%d | Train: %.4f | Val: %.4f | FP: V=%.3f iL=%.3f | LR=%.1e | %.1fs | %.2fh%s\n', ...
                ph.name,epoch,nEp,trainHist(end),vl,fpv,fpi,lr,epTime,elapsed,imp);
            if noImp>=EARLY_STOP&&~opts.test, fprintf('  Early stop\n'); break; end
        else
            if mod(epoch,50)==0, fprintf('  [%s] Ep %3d/%d | Train: %.4f | %.1fs\n',ph.name,epoch,nEp,trainHist(end),epTime); end
        end
        if toc(tLastCp)>CP_MIN*60, saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns); tLastCp=tic; end
        if toc(tStart)/3600>=opts.maxHours, fprintf('  Time limit\n'); break; end
    end
    saveCp(CP_DIR,sprintf('after_%s.mat',ph.name),net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
    if toc(tStart)/3600>=opts.maxHours, break; end
end

%% ── Phase 4: Extended ──────────────────────────────────────────────────────
if ~opts.test && toc(tStart)/3600<opts.maxHours
    fprintf('\n--- Phase4_Extended ---\n');
    if bestVal<inf, net=bestNet; end
    clearCache(accLoss);
    ext_lr=2e-4; ext_lr_min=1e-6;
    avgG=[]; avgSqG=[]; phIter=0; noImpExt=0; curLR=ext_lr; patCount=0; epCnt=0;

    while true
        epCnt=epCnt+1; totalEpochs=totalEpochs+1; tEp=tic;
        if epCnt<=200, winMs=20; elseif epCnt<=400, winMs=40; else, winMs=80; end
        winSamp=round(winMs*1e-3/TS);

        allWin=collectWindows(train_u,train_y,nTrain,winSamp,3);
        epochLoss=0; nWin=0;
        for w=1:size(allWin,1)
            phIter=phIter+1;
            [loss,grads]=dlfeval(accLoss,net,dlarray(allWin{w,2}(1,:)'),dlarray(allWin{w,1}),dlarray(allWin{w,2}),ns,DT,SKIP,0.1,b_lp,a_lp,b_hp,a_hp);
            if isnan(extractdata(loss))||isinf(extractdata(loss)), continue; end
            grads=clipGrads(grads,GRAD_CLIP);
            [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,phIter,curLR);
            epochLoss=epochLoss+extractdata(loss); nWin=nWin+1;
        end
        trainHist(end+1)=epochLoss/max(nWin,1); elapsed=toc(tStart)/3600; %#ok<AGROW>
        if elapsed>=opts.maxHours, fprintf('  [Ext] Time limit\n'); break; end
        if curLR<=ext_lr_min*1.01, fprintf('  [Ext] LR floor\n'); break; end

        if mod(epCnt,VAL_FREQ)==0||epCnt==1
            [vl,rv,ri]=computeVal(net,val_u,val_y,ns,DT,SKIP);
            [fpv,fpi]=computeFP(net,val_u,val_y,ns,DT,SKIP);
            valHist(end+1)=vl; imp=''; %#ok<AGROW>
            if vl<bestVal*(1-1e-4)
                bestVal=vl;bestNet=net;noImpExt=0;patCount=0;imp=' ** BEST **';
                saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
            else
                noImpExt=noImpExt+1; patCount=patCount+1;
                if patCount>=10, curLR=max(curLR*0.5,ext_lr_min); patCount=0;
                    fprintf('    LR->%.1e\n',curLR); end
            end
            fprintf('  [Ext ep%4d] Train:%.4f|Val:%.4f|FP:V=%.3f iL=%.3f|win=%dms|LR=%.1e|%.2fh|noImp=%d/30%s\n',...
                epCnt,trainHist(end),vl,fpv,fpi,winMs,curLR,elapsed,noImpExt,imp);
            if noImpExt>=30, fprintf('  [Ext] Plateau\n'); break; end
        end
        if toc(tLastCp)>CP_MIN*60, saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns); tLastCp=tic; end
    end
end

%% ── Final ──────────────────────────────────────────────────────────────────
totalH=toc(tStart)/3600;
if bestVal<inf, net=bestNet; end
[fpv,fpi]=computeFP(net,val_u,val_y,ns,DT,SKIP);
saveCp(CP_DIR,'final.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
fprintf('\n=== LPV Seed %d Complete | %.2fh | %d ep | bestVal=%.6f | FP: V=%.4f iL=%.4f ===\n',...
    seed,totalH,totalEpochs,bestVal,fpv,fpi);
fprintf('  PyTorch ref: Vout=0.019V iL=0.126A\n');
end


%% ========================================================================
function allWin = collectWindows(train_u, train_y, nTrain, winSamp, wpp)
allWin={}; perm=randperm(nTrain);
for p=perm
    u_p=train_u{p}; y_p=train_y{p}; T=size(u_p,1);
    nW=max(1,floor(T/winSamp)); nUse=min(wpp,nW);
    for s=randperm(nW,nUse)-1
        si=s*winSamp+1;
        if si+winSamp-1>T, continue; end
        allWin{end+1,1}=u_p(si:si+winSamp-1); allWin{end,2}=y_p(si:si+winSamp-1,:); %#ok<AGROW>
    end
end
allWin=allWin(randperm(size(allWin,1)),:);
end


%% ── LPV loss + gradient ────────────────────────────────────────────────────
function [loss,grads] = computeLPVLossAndGrad(net, x0, u_win, y_win, ns, dt, skip, lam_rip, b_lp, a_lp, b_hp, a_hp)
x0=stripdims(x0); u_win=stripdims(u_win); y_win=stripdims(y_win);
x_pred = rk4LPV(net, x0, u_win, ns, dt, skip);
T_raw=size(u_win,1); T_sub=size(x_pred,2);
idx=1:skip:T_raw; idx=idx(1:T_sub);
y_sub=y_win(idx,:)';
x_pred_n=(x_pred-ns.x_mean)./ns.x_std;
y_sub_n=(y_sub-ns.x_mean)./ns.x_std;

if lam_rip>0 && ~isempty(b_lp) && T_sub>20
    sk=max(floor(T_sub/10),5);
    xpT=x_pred_n'; ysT=y_sub_n';
    x_lp=applyIIR(xpT,b_lp,a_lp); y_lp=applyIIR(ysT,b_lp,a_lp);
    x_hp=applyIIR(xpT,b_hp,a_hp); y_hp=applyIIR(ysT,b_hp,a_hp);
    loss=mean((x_lp(sk+1:end,:)-y_lp(sk+1:end,:)).^2,'all')+lam_rip*mean((x_hp(sk+1:end,:)-y_hp(sk+1:end,:)).^2,'all');
else
    loss=mean((x_pred_n-y_sub_n).^2,'all');
end
grads=dlgradient(loss,net.Learnables);
end


%% ── LPV RK4 ────────────────────────────────────────────────────────────────
function x_traj = rk4LPV(net, x0, u_win, ns, dt, skip)
T_raw=size(u_win,1); T_sub=floor((T_raw-1)/skip)+1;
x=x0; x_traj=x;
for k=1:(T_sub-1)
    oi=min((k-1)*skip+1,T_raw); d_k=u_win(oi);
    k1=lpvFwd(net,x,d_k,ns); k2=lpvFwd(net,x+.5*dt*k1,d_k,ns);
    k3=lpvFwd(net,x+.5*dt*k2,d_k,ns); k4=lpvFwd(net,x+dt*k3,d_k,ns);
    x=x+(dt/6)*(k1+2*k2+2*k3+k4);
    x_traj=cat(2,x_traj,x);
end
end

function dxdt = lpvFwd(net, x, d, ns)
x_n = (x - ns.x_mean) ./ ns.x_std;
d_n = (d - ns.u_mean) / ns.u_std;
inp = dlarray([stripdims(x_n); d_n], 'CB');
raw = stripdims(forward(net, inp));  % [8x1]
% A: raw(1:4) reshaped [2x2] — transpose for row-major (PyTorch) to col-major (MATLAB)
A = reshape(raw(1:4), [2,2])';
B = reshape(raw(5:6), [2,1]);
c = raw(7:8);
dxdt_n = A * stripdims(x_n) + B * d_n + c;
dxdt = dxdt_n .* ns.dxdt_scale;
end


%% ── LPV Jacobian loss ──────────────────────────────────────────────────────
function [lJ,gJ] = computeLPVJacLoss(net, A_tgt, x_ops, u_ops, ns, lam_J)
nPts=size(A_tgt,1);
if nPts>6, fi=round(linspace(1,nPts,4)); rest=setdiff(1:nPts,fi);
    ri=rest(randperm(numel(rest),min(2,numel(rest)))); idx=[fi,ri];
else, idx=1:nPts; end

lJ=dlarray(single(0));
for j=idx
    xn=(x_ops(j,:)'-ns.x_mean)./ns.x_std;
    dn=(u_ops(j)-ns.u_mean)/ns.u_std;
    inp=dlarray([xn;dn],'CB');
    raw=stripdims(forward(net,inp));
    A_pred=reshape(raw(1:4),[2,2])';
    A_ref=dlarray(single(squeeze(A_tgt(j,:,:))'));
    A_sc=max(abs(extractdata(A_ref)),[],'all')+1;
    lJ=lJ+mean(((A_pred-A_ref)/A_sc).^2,'all');
end
lJ=lJ*(lam_J/numel(idx));
gJ=dlgradient(lJ,net.Learnables);
end


%% ── Shared helpers ──────────────────────────────────────────────────────────
function y=applyIIR(x,b,a)
[T,C]=size(x); order=numel(a)-1;
y=zeros(T,C,'like',x); d=zeros(order,C,'like',x);
for t=1:T
    y(t,:)=b(1)*x(t,:)+d(1,:);
    for i=1:order-1, d(i,:)=b(i+1)*x(t,:)-a(i+1)*y(t,:)+d(i+1,:); end
    d(order,:)=b(order+1)*x(t,:)-a(order+1)*y(t,:);
end
end

function [vl,rv,ri]=computeVal(net,vu,vy,ns,dt,sk)
vl=0;tv=0;ti=0;tn=0;
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    se=(extractdata(xp)-ys).^2;tv=tv+sum(se(1,:));ti=ti+sum(se(2,:));tn=tn+Ts;
end
rv=sqrt(tv/tn);ri=sqrt(ti/tn);
vl=(tv/ns.x_std(1)^2+ti/ns.x_std(2)^2)/(2*tn);
end

function [fpv,fpi]=computeFP(net,vu,vy,ns,dt,sk)
rv=[];ri=[];
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    rv(end+1)=sqrt(mean((extractdata(xp(1,:))-ys(1,:)).^2)); %#ok<AGROW>
    ri(end+1)=sqrt(mean((extractdata(xp(2,:))-ys(2,:)).^2)); %#ok<AGROW>
end
fpv=mean(rv);fpi=mean(ri);
end

function g=clipGrads(g,maxN)
sq=0;for i=1:height(g),sq=sq+sum(extractdata(g.Value{i}).^2,'all');end
n=sqrt(sq); if n>maxN, s=maxN/(n+1e-8);for i=1:height(g),g.Value{i}=g.Value{i}*s;end;end
end

function saveCp(d,f,net,bestNet,bestVal,th,vh,te,ns)
fp=fullfile(d,f); save(fp,'net','bestNet','bestVal','th','vh','te','ns','-v7.3');
fprintf('    [cp] %s (val=%.4f)\n',f,bestVal);
end
