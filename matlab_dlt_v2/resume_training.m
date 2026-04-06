function resume_training(arch, seed, opts)
% resume_training  Resume MLP or LPV training from best checkpoint.
%
% Loads best.mat from previous run, runs Phase 3 + Phase 4 extended.
% Saves to the SAME run folder (run_NNNN/) — best.mat updated only if improved.
%
% Usage:
%   resume_training('lpv', 7)                    % resume LPV seed 7
%   resume_training('mlp', 3)                    % resume MLP seed 3
%   resume_training('lpv', 7, 'maxHours', 8)     % with time limit
%
% Phases run:
%   Phase 2 remainder (if startPhase=2): remaining epochs of Phase 2
%   Phase 3: 200 ep, 20ms, LR=1e-4, cosine annealing, ripple loss
%   Phase 4: Extended, ReduceLROnPlateau, progressive windows 20->40->80ms

arguments
    arch (1,:) char {mustBeMember(arch, {'mlp','lpv'})}
    seed (1,1) double
    opts.maxHours (1,1) double = 8
    opts.startPhase (1,1) double = 3   % 2 = resume Phase 2, 3 = start Phase 3
    opts.startEpoch (1,1) double = 1   % starting epoch within startPhase
end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

TS=5e-6; SKIP=10; DT=TS*SKIP; NX=2; H=64; GRAD_CLIP=1.0;
VAL_FREQ=10; CP_MIN=15;
VAL_IDX=[5,12,14];

if strcmp(arch,'mlp')
    CP_DIR = fullfile(thisDir, 'checkpoints_mlp', sprintf('run_%04d', seed));
    isLPV = false;
else
    CP_DIR = fullfile(thisDir, 'checkpoints_lpv', sprintf('run_%04d', seed));
    isLPV = true;
end

maxNumCompThreads(1);

%% ── Load best checkpoint ───────────────────────────────────────────────────
bestFile = fullfile(CP_DIR, 'best.mat');
if ~isfile(bestFile)
    error('No best.mat found in %s', CP_DIR);
end
cp = load(bestFile);
net = cp.bestNet;
bestVal = cp.bestVal;
ns = cp.ns;
% Handle both long and short field names (MLP uses long, LPV uses short)
if isfield(cp, 'trainHist'), trainHist = cp.trainHist; valHist = cp.valHist; totalEpochs = cp.totalEpochs;
elseif isfield(cp, 'th'), trainHist = cp.th; valHist = cp.vh; totalEpochs = cp.te;
else, trainHist = []; valHist = []; totalEpochs = 0; end

fprintf('=== Resume %s Seed %d ===\n', upper(arch), seed);
fprintf('  Loaded best.mat: val=%.6f, epochs=%d\n', bestVal, totalEpochs);
fprintf('  FP target: %s\n', ternary(isLPV, 'Vout=0.019V (PyTorch LPV)', 'Vout=0.036V (PyTorch MLP)'));
fprintf('  maxHours=%.1f\n', opts.maxHours);

%% ── Load data ──────────────────────────────────────────────────────────────
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
train_u = profiles_u(TRAIN_IDX); train_y = profiles_y(TRAIN_IDX);
val_u = profiles_u(VAL_IDX); val_y = profiles_y(VAL_IDX);

%% ── Filters ────────────────────────────────────────────────────────────────
f_cut=sqrt(600*200e3); fs_eff=1/DT; fn=min(f_cut/(fs_eff/2),0.95);
if fn>0.05
    [b_lp,a_lp]=butter(2,fn,'low'); [b_hp,a_hp]=butter(2,fn,'high');
    b_lp=single(b_lp);a_lp=single(a_lp);b_hp=single(b_hp);a_hp=single(a_hp);
else, b_lp=[];a_lp=[];b_hp=[];a_hp=[]; end

%% ── Choose forward function ────────────────────────────────────────────────
if isLPV
    accLoss = dlaccelerate(@computeLPVLossAndGrad);
    valFn = @(n,vu,vy) computeValLPV(n,vu,vy,ns,DT,SKIP);
    fpFn  = @(n,vu,vy) computeFPLPV(n,vu,vy,ns,DT,SKIP);
else
    accLoss = dlaccelerate(@computeMLPLossAndGrad);
    valFn = @(n,vu,vy) computeValMLP(n,vu,vy,ns,DT,SKIP);
    fpFn  = @(n,vu,vy) computeFPMLP(n,vu,vy,ns,DT,SKIP);
end

bestNet = net;
tStart = tic; tLastCp = tic;

%% ── Phase 2 remainder (if startPhase=2) ────────────────────────────────────
if opts.startPhase <= 2
    nEp2 = 500; startEp2 = opts.startEpoch;
    remaining2 = nEp2 - startEp2 + 1;
    if remaining2 > 0
        winSamp = round(20e-3/TS); lr_max=5e-4; lr_min=lr_max*0.01;
        clearCache(accLoss);
        fprintf('\n--- Phase2_FullOsc (resume): ep %d->%d | win=20ms | LR=%.1e ---\n', startEp2, nEp2, lr_max);

        avgG=[]; avgSqG=[]; phIter=0; noImp=0;
        % Load Jacobian targets for Phase 2
        jacFile = fullfile(repoRoot,'data','neural_ode','jacobian_targets.json');
        hasJac=false; A_tgt=[];x_ops=[];u_ops=[];
        if isfile(jacFile)
            jd=jsondecode(fileread(jacFile));
            A_tgt=single(jd.A_targets);x_ops=single(jd.x_ops);u_ops=single(jd.u_ops(:));
            if ndims(A_tgt)==3&&size(A_tgt,1)==2,A_tgt=permute(A_tgt,[3 1 2]);end
            hasJac=true;
        end

        for epoch = startEp2:nEp2
            totalEpochs=totalEpochs+1; tEp=tic;
            lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(epoch-1)/nEp2));

            allWin = collectWindows(train_u,train_y,nTrain,winSamp,3);
            epochLoss=0; nWin=0;
            for w=1:size(allWin,1)
                x0=dlarray(allWin{w,2}(1,:)');
                phIter=phIter+1;
                [loss,grads]=dlfeval(accLoss,net,x0,dlarray(allWin{w,1}),dlarray(allWin{w,2}),ns,DT,SKIP,0.1,b_lp,a_lp,b_hp,a_hp);
                if isnan(extractdata(loss))||isinf(extractdata(loss)),continue;end
                grads=clipGrads(grads,GRAD_CLIP);
                [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,phIter,lr);
                epochLoss=epochLoss+extractdata(loss);nWin=nWin+1;

                if hasJac && 0.01>0 && mod(nWin,5)==0
                    phIter=phIter+1;
                    if isLPV
                        [lJ,gJ]=dlfeval(@computeLPVJacLoss,net,A_tgt,x_ops,u_ops,ns,0.01);
                    else
                        [lJ,gJ]=dlfeval(@computeMLPJacLoss,net,A_tgt,x_ops,u_ops,ns,0.01);
                    end
                    if ~isnan(extractdata(lJ))
                        gJ=clipGrads(gJ,GRAD_CLIP);
                        [net,avgG,avgSqG]=adamupdate(net,gJ,avgG,avgSqG,phIter,lr);
                    end
                end
            end
            trainHist(end+1)=epochLoss/max(nWin,1);epTime=toc(tEp); %#ok<AGROW>

            if mod(epoch,VAL_FREQ)==0||epoch==startEp2||epoch==nEp2
                [vl,rv,ri]=valFn(net,val_u,val_y);
                [fpv,fpi]=fpFn(net,val_u,val_y);
                valHist(end+1)=vl;imp=''; %#ok<AGROW>
                if vl<bestVal, bestVal=vl;bestNet=net;noImp=0;imp=' ** BEST **';
                    saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
                else,noImp=noImp+1;end
                elapsed=toc(tStart)/3600;
                fprintf('  [Ph2] Ep %3d/%d|Train:%.4f|Val:%.4f|FP:V=%.3f iL=%.3f|LR=%.1e|%.1fs|%.2fh%s\n',...
                    epoch,nEp2,trainHist(end),vl,fpv,fpi,lr,epTime,elapsed,imp);
                if noImp>=20,fprintf('  Early stop\n');break;end
            end
            if toc(tLastCp)>CP_MIN*60,saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);tLastCp=tic;end
            if toc(tStart)/3600>=opts.maxHours,fprintf('  Time limit\n');break;end
        end
        saveCp(CP_DIR,'after_Phase2_resume.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
        fprintf('  Phase2 remainder done. Best val: %.6f\n', bestVal);
    end
end

if toc(tStart)/3600>=opts.maxHours
    fprintf('Time limit reached before Phase 3\n');
    net=bestNet;[fpv,fpi]=fpFn(net,val_u,val_y);
    saveCp(CP_DIR,'final_resume.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
    fprintf('\n=== Resume %s Seed %d | %.2fh | bestVal=%.6f | FP:V=%.4f iL=%.4f ===\n',upper(arch),seed,toc(tStart)/3600,bestVal,fpv,fpi);
    return;
end

%% ── Phase 3: Finetune ─────────────────────────────────────────────────────
fprintf('\n--- Phase3_Finetune: 200 ep | win=20ms | LR=1e-4->1e-6 ---\n');
net = bestNet;  % start Phase 3 from best model
winSamp = round(20e-3/TS); lr_max = 1e-4; lr_min = lr_max*0.01; nEp = 200;
clearCache(accLoss);
avgG=[]; avgSqG=[]; phIter=0; noImp=0;

for epoch = 1:nEp
    totalEpochs = totalEpochs+1; tEp = tic;
    lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(epoch-1)/nEp));

    allWin = collectWindows(train_u, train_y, nTrain, winSamp, 3);
    epochLoss=0; nWin=0;
    for w = 1:size(allWin,1)
        x0 = dlarray(allWin{w,2}(1,:)');
        phIter = phIter+1;
        [loss,grads] = dlfeval(accLoss, net, x0, dlarray(allWin{w,1}), ...
            dlarray(allWin{w,2}), ns, DT, SKIP, 0.1, b_lp, a_lp, b_hp, a_hp);
        if isnan(extractdata(loss))||isinf(extractdata(loss)), continue; end
        grads = clipGrads(grads, GRAD_CLIP);
        [net,avgG,avgSqG] = adamupdate(net, grads, avgG, avgSqG, phIter, lr);
        epochLoss = epochLoss+extractdata(loss); nWin = nWin+1;
    end
    trainHist(end+1) = epochLoss/max(nWin,1); epTime = toc(tEp); %#ok<AGROW>

    if mod(epoch,VAL_FREQ)==0 || epoch==1 || epoch==nEp
        [vl,rv,ri] = valFn(net, val_u, val_y);
        [fpv,fpi] = fpFn(net, val_u, val_y);
        valHist(end+1) = vl; imp=''; %#ok<AGROW>
        if vl < bestVal
            bestVal=vl; bestNet=net; noImp=0; imp=' ** BEST **';
            saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
        else, noImp=noImp+1; end
        elapsed = toc(tStart)/3600;
        fprintf('  [Ph3] Ep %3d/%d | Train:%.4f | Val:%.4f | FP:V=%.3f iL=%.3f | LR=%.1e | %.1fs | %.2fh%s\n', ...
            epoch,nEp,trainHist(end),vl,fpv,fpi,lr,epTime,elapsed,imp);
        if noImp>=20, fprintf('  Early stop\n'); break; end
    end
    if toc(tLastCp)>CP_MIN*60, saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns); tLastCp=tic; end
    if toc(tStart)/3600>=opts.maxHours, fprintf('  Time limit\n'); break; end
end
saveCp(CP_DIR,'after_Phase3_resume.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
fprintf('  Phase3 done. Best val: %.6f\n', bestVal);

%% ── Phase 4: Extended ──────────────────────────────────────────────────────
if toc(tStart)/3600 < opts.maxHours
    fprintf('\n--- Phase4_Extended: ReduceLROnPlateau, progressive windows ---\n');
    net = bestNet; clearCache(accLoss);
    ext_lr=2e-4; ext_lr_min=1e-6;
    avgG=[]; avgSqG=[]; phIter=0; noImpExt=0; curLR=ext_lr; patCount=0; epCnt=0;

    while true
        epCnt=epCnt+1; totalEpochs=totalEpochs+1; tEp=tic;
        if epCnt<=200, winMs=20; elseif epCnt<=400, winMs=40; else, winMs=80; end
        winSamp = round(winMs*1e-3/TS);

        allWin = collectWindows(train_u, train_y, nTrain, winSamp, 3);
        epochLoss=0; nWin=0;
        for w = 1:size(allWin,1)
            phIter=phIter+1;
            [loss,grads] = dlfeval(accLoss, net, dlarray(allWin{w,2}(1,:)'), ...
                dlarray(allWin{w,1}), dlarray(allWin{w,2}), ns, DT, SKIP, ...
                0.1, b_lp, a_lp, b_hp, a_hp);
            if isnan(extractdata(loss))||isinf(extractdata(loss)), continue; end
            grads = clipGrads(grads, GRAD_CLIP);
            [net,avgG,avgSqG] = adamupdate(net, grads, avgG, avgSqG, phIter, curLR);
            epochLoss = epochLoss+extractdata(loss); nWin = nWin+1;
        end
        trainHist(end+1) = epochLoss/max(nWin,1); elapsed=toc(tStart)/3600; %#ok<AGROW>
        if elapsed>=opts.maxHours, fprintf('  [Ext] Time limit\n'); break; end
        if curLR<=ext_lr_min*1.01, fprintf('  [Ext] LR floor\n'); break; end

        if mod(epCnt,VAL_FREQ)==0 || epCnt==1
            [vl,rv,ri] = valFn(net, val_u, val_y);
            [fpv,fpi] = fpFn(net, val_u, val_y);
            valHist(end+1) = vl; imp=''; %#ok<AGROW>
            if vl < bestVal*(1-1e-4)
                bestVal=vl; bestNet=net; noImpExt=0; patCount=0; imp=' ** BEST **';
                saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
            else
                noImpExt=noImpExt+1; patCount=patCount+1;
                if patCount>=10, curLR=max(curLR*0.5,ext_lr_min); patCount=0;
                    fprintf('    LR->%.1e\n',curLR); end
            end
            fprintf('  [Ext ep%4d] Train:%.4f|Val:%.4f|FP:V=%.3f iL=%.3f|win=%dms|LR=%.1e|%.2fh|noImp=%d/60%s\n',...
                epCnt,trainHist(end),vl,fpv,fpi,winMs,curLR,elapsed,noImpExt,imp);
            if noImpExt>=60, fprintf('  [Ext] Plateau\n'); break; end  % 60 instead of 30 — more patient
        end
        if toc(tLastCp)>CP_MIN*60, saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns); tLastCp=tic; end
    end
end

%% ── Final ──────────────────────────────────────────────────────────────────
totalH = toc(tStart)/3600;
net = bestNet;
[fpv,fpi] = fpFn(net, val_u, val_y);
saveCp(CP_DIR,'final_resume.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
fprintf('\n=== Resume %s Seed %d Complete | %.2fh | %d ep | bestVal=%.6f | FP:V=%.4f iL=%.4f ===\n',...
    upper(arch),seed,totalH,totalEpochs,bestVal,fpv,fpi);
end


%% ========================================================================
%% Shared helpers
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

function g=clipGrads(g,maxN)
sq=0;for i=1:height(g),sq=sq+sum(extractdata(g.Value{i}).^2,'all');end
n=sqrt(sq);if n>maxN,s=maxN/(n+1e-8);for i=1:height(g),g.Value{i}=g.Value{i}*s;end;end
end

function saveCp(d,f,net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns)
fp=fullfile(d,f);save(fp,'net','bestNet','bestVal','trainHist','valHist','totalEpochs','ns','-v7.3');
fprintf('    [cp] %s (val=%.4f)\n',f,bestVal);
end

function s = ternary(cond, a, b)
if cond, s=a; else, s=b; end
end


%% ========================================================================
%% MLP forward/val/fp
%% ========================================================================
function [loss,grads] = computeMLPLossAndGrad(net, x0, u_win, y_win, ns, dt, skip, lam_rip, b_lp, a_lp, b_hp, a_hp)
x0=stripdims(x0); u_win=stripdims(u_win); y_win=stripdims(y_win);
x_pred=rk4MLP(net,x0,u_win,ns,dt,skip);
T_raw=size(u_win,1); T_sub=size(x_pred,2);
idx=1:skip:T_raw; idx=idx(1:T_sub); y_sub=y_win(idx,:)';
xn=(x_pred-ns.x_mean)./ns.x_std; yn=(y_sub-ns.x_mean)./ns.x_std;
if lam_rip>0&&~isempty(b_lp)&&T_sub>20
    sk=max(floor(T_sub/10),5);
    xl=applyIIR(xn',b_lp,a_lp);yl=applyIIR(yn',b_lp,a_lp);
    xh=applyIIR(xn',b_hp,a_hp);yh=applyIIR(yn',b_hp,a_hp);
    loss=mean((xl(sk+1:end,:)-yl(sk+1:end,:)).^2,'all')+lam_rip*mean((xh(sk+1:end,:)-yh(sk+1:end,:)).^2,'all');
else, loss=mean((xn-yn).^2,'all'); end
grads=dlgradient(loss,net.Learnables);
end

function x_traj=rk4MLP(net,x0,u,ns,dt,sk)
T=size(u,1);Ts=floor((T-1)/sk)+1;x=x0;x_traj=x;
for k=1:Ts-1
    oi=min((k-1)*sk+1,T);d=u(oi);
    k1=mlpF(net,x,d,ns);k2=mlpF(net,x+.5*dt*k1,d,ns);
    k3=mlpF(net,x+.5*dt*k2,d,ns);k4=mlpF(net,x+dt*k3,d,ns);
    x=x+(dt/6)*(k1+2*k2+2*k3+k4);x_traj=cat(2,x_traj,x);
end
end

function dxdt=mlpF(net,x,d,ns)
xn=(x-ns.x_mean)./ns.x_std; dn=(d-ns.u_mean)/ns.u_std;
inp=dlarray([stripdims(xn);dn],'CB');
dxdt=stripdims(forward(net,inp)).*ns.dxdt_scale;
end

function [vl,rv,ri]=computeValMLP(net,vu,vy,ns,dt,sk)
tv=0;ti=0;tn=0;
for v=1:numel(vu)
    xp=rk4MLP(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    se=(extractdata(xp)-ys).^2;tv=tv+sum(se(1,:));ti=ti+sum(se(2,:));tn=tn+Ts;
end
rv=sqrt(tv/tn);ri=sqrt(ti/tn);vl=(tv/ns.x_std(1)^2+ti/ns.x_std(2)^2)/(2*tn);
end

function [fpv,fpi]=computeFPMLP(net,vu,vy,ns,dt,sk)
rv=[];ri=[];
for v=1:numel(vu)
    xp=rk4MLP(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    rv(end+1)=sqrt(mean((extractdata(xp(1,:))-ys(1,:)).^2));ri(end+1)=sqrt(mean((extractdata(xp(2,:))-ys(2,:)).^2)); %#ok<AGROW>
end
fpv=mean(rv);fpi=mean(ri);
end


%% ========================================================================
%% LPV forward/val/fp
%% ========================================================================
function [loss,grads] = computeLPVLossAndGrad(net, x0, u_win, y_win, ns, dt, skip, lam_rip, b_lp, a_lp, b_hp, a_hp)
x0=stripdims(x0); u_win=stripdims(u_win); y_win=stripdims(y_win);
x_pred=rk4LPV(net,x0,u_win,ns,dt,skip);
T_raw=size(u_win,1); T_sub=size(x_pred,2);
idx=1:skip:T_raw; idx=idx(1:T_sub); y_sub=y_win(idx,:)';
xn=(x_pred-ns.x_mean)./ns.x_std; yn=(y_sub-ns.x_mean)./ns.x_std;
if lam_rip>0&&~isempty(b_lp)&&T_sub>20
    sk=max(floor(T_sub/10),5);
    xl=applyIIR(xn',b_lp,a_lp);yl=applyIIR(yn',b_lp,a_lp);
    xh=applyIIR(xn',b_hp,a_hp);yh=applyIIR(yn',b_hp,a_hp);
    loss=mean((xl(sk+1:end,:)-yl(sk+1:end,:)).^2,'all')+lam_rip*mean((xh(sk+1:end,:)-yh(sk+1:end,:)).^2,'all');
else, loss=mean((xn-yn).^2,'all'); end
grads=dlgradient(loss,net.Learnables);
end

function x_traj=rk4LPV(net,x0,u,ns,dt,sk)
T=size(u,1);Ts=floor((T-1)/sk)+1;x=x0;x_traj=x;
for k=1:Ts-1
    oi=min((k-1)*sk+1,T);d=u(oi);
    k1=lpvF(net,x,d,ns);k2=lpvF(net,x+.5*dt*k1,d,ns);
    k3=lpvF(net,x+.5*dt*k2,d,ns);k4=lpvF(net,x+dt*k3,d,ns);
    x=x+(dt/6)*(k1+2*k2+2*k3+k4);x_traj=cat(2,x_traj,x);
end
end

function dxdt=lpvF(net,x,d,ns)
xn=(x-ns.x_mean)./ns.x_std; dn=(d-ns.u_mean)/ns.u_std;
inp=dlarray([stripdims(xn);dn],'CB');
raw=stripdims(forward(net,inp));
A=reshape(raw(1:4),[2,2])'; B=reshape(raw(5:6),[2,1]); c=raw(7:8);
dxdt=(A*stripdims(xn)+B*dn+c).*ns.dxdt_scale;
end

function [vl,rv,ri]=computeValLPV(net,vu,vy,ns,dt,sk)
tv=0;ti=0;tn=0;
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    se=(extractdata(xp)-ys).^2;tv=tv+sum(se(1,:));ti=ti+sum(se(2,:));tn=tn+Ts;
end
rv=sqrt(tv/tn);ri=sqrt(ti/tn);vl=(tv/ns.x_std(1)^2+ti/ns.x_std(2)^2)/(2*tn);
end

function [fpv,fpi]=computeFPLPV(net,vu,vy,ns,dt,sk)
rv=[];ri=[];
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    rv(end+1)=sqrt(mean((extractdata(xp(1,:))-ys(1,:)).^2));ri(end+1)=sqrt(mean((extractdata(xp(2,:))-ys(2,:)).^2)); %#ok<AGROW>
end
fpv=mean(rv);fpi=mean(ri);
end

function y=applyIIR(x,b,a)
[T,C]=size(x);order=numel(a)-1;
y=zeros(T,C,'like',x);d=zeros(order,C,'like',x);
for t=1:T
    y(t,:)=b(1)*x(t,:)+d(1,:);
    for i=1:order-1,d(i,:)=b(i+1)*x(t,:)-a(i+1)*y(t,:)+d(i+1,:);end
    d(order,:)=b(order+1)*x(t,:)-a(order+1)*y(t,:);
end
end

%% ── Jacobian losses ────────────────────────────────────────────────────────
function [lJ,gJ]=computeMLPJacLoss(net,A_tgt,x_ops,u_ops,ns,lam_J)
W1=net.Learnables.Value{1};b1=net.Learnables.Value{2};
W2=net.Learnables.Value{3};b2=net.Learnables.Value{4};
W3=net.Learnables.Value{5};
nPts=size(A_tgt,1);
if nPts>6,fi=round(linspace(1,nPts,4));rest=setdiff(1:nPts,fi);
    ri=rest(randperm(numel(rest),min(2,numel(rest))));idx=[fi,ri];
else,idx=1:nPts;end
lJ=dlarray(single(0));
for j=idx
    xn=(x_ops(j,:)'-ns.x_mean)./ns.x_std; dn=(u_ops(j)-ns.u_mean)/ns.u_std;
    inp=[xn;dn]; z1=W1*inp+b1(:);a1=tanh(z1);z2=W2*a1+b2(:);a2=tanh(z2);
    D1=1-a1.^2;D2=1-a2.^2;
    J_phys=ns.dxdt_scale.*(W3*(D2.*(W2*(D1.*W1(:,1:2)))))./ns.x_std';
    A_ref=dlarray(single(squeeze(A_tgt(j,:,:))'));
    A_sc=max(abs(extractdata(A_ref)),[],'all')+1;
    lJ=lJ+mean(((J_phys-A_ref)/A_sc).^2,'all');
end
lJ=lJ*(lam_J/numel(idx)); gJ=dlgradient(lJ,net.Learnables);
end

function [lJ,gJ]=computeLPVJacLoss(net,A_tgt,x_ops,u_ops,ns,lam_J)
nPts=size(A_tgt,1);
if nPts>6,fi=round(linspace(1,nPts,4));rest=setdiff(1:nPts,fi);
    ri=rest(randperm(numel(rest),min(2,numel(rest))));idx=[fi,ri];
else,idx=1:nPts;end
lJ=dlarray(single(0));
for j=idx
    xn=(x_ops(j,:)'-ns.x_mean)./ns.x_std;dn=(u_ops(j)-ns.u_mean)/ns.u_std;
    inp=dlarray([xn;dn],'CB');raw=stripdims(forward(net,inp));
    A_pred=reshape(raw(1:4),[2,2])';
    A_ref=dlarray(single(squeeze(A_tgt(j,:,:))'));
    A_sc=max(abs(extractdata(A_ref)),[],'all')+1;
    lJ=lJ+mean(((A_pred-A_ref)/A_sc).^2,'all');
end
lJ=lJ*(lam_J/numel(idx));gJ=dlgradient(lJ,net.Learnables);
end
