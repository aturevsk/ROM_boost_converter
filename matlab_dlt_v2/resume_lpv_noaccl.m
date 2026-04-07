function resume_lpv_noaccl(maxHours)
% Resume LPV seed 7 from best.mat, Phase 4 extended at 40ms+ windows.
% NO dlaccelerate — avoids retrace spikes on long windows.
%
% Usage: resume_lpv_noaccl(8)

if nargin < 1, maxHours = 8; end

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(genpath(repoRoot));

TS=5e-6; SKIP=10; DT=TS*SKIP; NX=2; GRAD_CLIP=1.0;
VAL_FREQ=10; CP_MIN=15;
VAL_IDX=[5,12,14];

CP_DIR = fullfile(thisDir, 'checkpoints_lpv', 'run_0007');
maxNumCompThreads(1);

%% ── Load best checkpoint ───────────────────────────────────────────────────
cp = load(fullfile(CP_DIR, 'best.mat'));
net = cp.bestNet; bestVal = cp.bestVal; ns = cp.ns;
if isfield(cp,'th'), trainHist=cp.th; valHist=cp.vh; totalEpochs=cp.te;
else, trainHist=cp.trainHist; valHist=cp.valHist; totalEpochs=cp.totalEpochs; end

fprintf('=== Resume LPV Seed 7 (NO dlaccelerate) ===\n');
fprintf('  Loaded: val=%.6f, epochs=%d\n', bestVal, totalEpochs);
fprintf('  maxHours=%.1f\n', maxHours);

%% ── Load data ──────────────────────────────────────────────────────────────
csvDir = fullfile(repoRoot,'data','neural_ode');
csvFiles = dir(fullfile(csvDir,'profile_*.csv'));
[~,ord]=sort({csvFiles.name}); csvFiles=csvFiles(ord);
nTotal=numel(csvFiles);
profiles_u=cell(nTotal,1); profiles_y=cell(nTotal,1);
for k=1:nTotal
    raw=single(readmatrix(fullfile(csvDir,csvFiles(k).name)));
    profiles_u{k}=raw(:,1); profiles_y{k}=raw(:,2:3);
end
TRAIN_IDX=setdiff(1:nTotal,VAL_IDX); nTrain=numel(TRAIN_IDX);
train_u=profiles_u(TRAIN_IDX); train_y=profiles_y(TRAIN_IDX);
val_u=profiles_u(VAL_IDX); val_y=profiles_y(VAL_IDX);

%% ── Filters ────────────────────────────────────────────────────────────────
f_cut=sqrt(600*200e3); fs_eff=1/DT; fn=min(f_cut/(fs_eff/2),0.95);
[b_lp,a_lp]=butter(2,fn,'low'); [b_hp,a_hp]=butter(2,fn,'high');
b_lp=single(b_lp);a_lp=single(a_lp);b_hp=single(b_hp);a_hp=single(a_hp);

%% ── Phase 4 Extended (40ms+ windows, no dlaccelerate) ──────────────────────
fprintf('\n--- Extended: 40ms->80ms windows, NO dlaccelerate ---\n');
bestNet=net; ext_lr=1e-4; ext_lr_min=1e-6;
avgG=[]; avgSqG=[]; phIter=0; noImpExt=0; curLR=ext_lr; patCount=0; epCnt=0;
tStart=tic; tLastCp=tic;

while true
    epCnt=epCnt+1; totalEpochs=totalEpochs+1; tEp=tic;
    winMs=80;
    winSamp=round(winMs*1e-3/TS);

    allWin=collectWindows(train_u,train_y,nTrain,winSamp,3);
    epochLoss=0; nWin=0;
    for w=1:size(allWin,1)
        x0=dlarray(allWin{w,2}(1,:)');
        phIter=phIter+1;
        % NO dlaccelerate — direct dlfeval
        [loss,grads]=dlfeval(@computeLoss, net, x0, dlarray(allWin{w,1}), ...
            dlarray(allWin{w,2}), ns, DT, SKIP, 0.1, b_lp, a_lp, b_hp, a_hp);
        if isnan(extractdata(loss))||isinf(extractdata(loss)), continue; end
        grads=clipGrads(grads,GRAD_CLIP);
        [net,avgG,avgSqG]=adamupdate(net,grads,avgG,avgSqG,phIter,curLR);
        epochLoss=epochLoss+extractdata(loss); nWin=nWin+1;
    end
    trainHist(end+1)=epochLoss/max(nWin,1); %#ok<AGROW>
    elapsed=toc(tStart)/3600; epTime=toc(tEp);

    if elapsed>=maxHours, fprintf('  Time limit\n'); break; end
    if curLR<=ext_lr_min*1.01, fprintf('  LR floor\n'); break; end

    if mod(epCnt,VAL_FREQ)==0 || epCnt==1
        [vl,rv,ri]=computeVal(net,val_u,val_y,ns,DT,SKIP);
        [fpv,fpi]=computeFP(net,val_u,val_y,ns,DT,SKIP);
        valHist(end+1)=vl; imp=''; %#ok<AGROW>
        if vl<bestVal*(1-1e-4)
            bestVal=vl; bestNet=net; noImpExt=0; patCount=0; imp=' ** BEST **';
            saveCp(CP_DIR,'best.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
        else
            noImpExt=noImpExt+1; patCount=patCount+1;
            if patCount>=10, curLR=max(curLR*0.5,ext_lr_min); patCount=0;
                fprintf('    LR->%.1e\n',curLR); end
        end
        fprintf('  [ep%4d] Train:%.4f|Val:%.4f|FP:V=%.3f iL=%.3f|win=%dms|LR=%.1e|%.1fs|%.2fh|noImp=%d/60%s\n',...
            epCnt,trainHist(end),vl,fpv,fpi,winMs,curLR,epTime,elapsed,noImpExt,imp);
        if noImpExt>=60, fprintf('  Plateau\n'); break; end
    else
        if mod(epCnt,5)==0, fprintf('  [ep%4d] Train:%.4f|win=%dms|%.1fs\n',epCnt,trainHist(end),winMs,epTime); end
    end
    if toc(tLastCp)>CP_MIN*60, saveCp(CP_DIR,'latest.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns); tLastCp=tic; end
end

%% ── Final ──────────────────────────────────────────────────────────────────
net=bestNet;
[fpv,fpi]=computeFP(net,val_u,val_y,ns,DT,SKIP);
saveCp(CP_DIR,'final_noaccl.mat',net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns);
fprintf('\n=== Complete | %.2fh | bestVal=%.6f | FP:V=%.4f iL=%.4f ===\n',...
    toc(tStart)/3600,bestVal,fpv,fpi);
end


%% ========================================================================
function allWin=collectWindows(tu,ty,nT,ws,wpp)
allWin={};perm=randperm(nT);
for p=perm
    u=tu{p};y=ty{p};T=size(u,1);nW=max(1,floor(T/ws));nU=min(wpp,nW);
    for s=randperm(nW,nU)-1
        si=s*ws+1;if si+ws-1>T,continue;end
        allWin{end+1,1}=u(si:si+ws-1);allWin{end,2}=y(si:si+ws-1,:); %#ok<AGROW>
    end
end
allWin=allWin(randperm(size(allWin,1)),:);
end

function [loss,grads]=computeLoss(net,x0,u_win,y_win,ns,dt,skip,lam_rip,b_lp,a_lp,b_hp,a_hp)
x0=stripdims(x0);u_win=stripdims(u_win);y_win=stripdims(y_win);
x_pred=rk4LPV(net,x0,u_win,ns,dt,skip);
T_raw=size(u_win,1);T_sub=size(x_pred,2);
idx=1:skip:T_raw;idx=idx(1:T_sub);y_sub=y_win(idx,:)';
xn=(x_pred-ns.x_mean)./ns.x_std;yn=(y_sub-ns.x_mean)./ns.x_std;
if lam_rip>0&&~isempty(b_lp)&&T_sub>20
    sk=max(floor(T_sub/10),5);
    xl=applyIIR(xn',b_lp,a_lp);yl=applyIIR(yn',b_lp,a_lp);
    xh=applyIIR(xn',b_hp,a_hp);yh=applyIIR(yn',b_hp,a_hp);
    loss=mean((xl(sk+1:end,:)-yl(sk+1:end,:)).^2,'all')+lam_rip*mean((xh(sk+1:end,:)-yh(sk+1:end,:)).^2,'all');
else,loss=mean((xn-yn).^2,'all');end
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
xn=(x-ns.x_mean)./ns.x_std;dn=(d-ns.u_mean)/ns.u_std;
inp=dlarray([stripdims(xn);dn],'CB');raw=stripdims(forward(net,inp));
A=reshape(raw(1:4),[2,2])';B=reshape(raw(5:6),[2,1]);c=raw(7:8);
dxdt=(A*stripdims(xn)+B*dn+c).*ns.dxdt_scale;
end

function y=applyIIR(x,b,a)
[T,C]=size(x);order=numel(a)-1;y=zeros(T,C,'like',x);d=zeros(order,C,'like',x);
for t=1:T,y(t,:)=b(1)*x(t,:)+d(1,:);
    for i=1:order-1,d(i,:)=b(i+1)*x(t,:)-a(i+1)*y(t,:)+d(i+1,:);end
    d(order,:)=b(order+1)*x(t,:)-a(order+1)*y(t,:);end
end

function [vl,rv,ri]=computeVal(net,vu,vy,ns,dt,sk)
tv=0;ti=0;tn=0;
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    se=(extractdata(xp)-ys).^2;tv=tv+sum(se(1,:));ti=ti+sum(se(2,:));tn=tn+Ts;
end
rv=sqrt(tv/tn);ri=sqrt(ti/tn);vl=(tv/ns.x_std(1)^2+ti/ns.x_std(2)^2)/(2*tn);
end

function [fpv,fpi]=computeFP(net,vu,vy,ns,dt,sk)
rv=[];ri=[];
for v=1:numel(vu)
    xp=rk4LPV(net,dlarray(single(vy{v}(1,:)')),dlarray(single(vu{v})),ns,dt,sk);
    Ts=size(xp,2);idx=1:sk:size(vu{v},1);idx=idx(1:Ts);ys=vy{v}(idx,:)';
    rv(end+1)=sqrt(mean((extractdata(xp(1,:))-ys(1,:)).^2));ri(end+1)=sqrt(mean((extractdata(xp(2,:))-ys(2,:)).^2)); %#ok<AGROW>
end
fpv=mean(rv);fpi=mean(ri);
end

function g=clipGrads(g,maxN)
sq=0;for i=1:height(g),sq=sq+sum(extractdata(g.Value{i}).^2,'all');end
n=sqrt(sq);if n>maxN,s=maxN/(n+1e-8);for i=1:height(g),g.Value{i}=g.Value{i}*s;end;end
end

function saveCp(d,f,net,bestNet,bestVal,trainHist,valHist,totalEpochs,ns)
fp=fullfile(d,f);save(fp,'net','bestNet','bestVal','trainHist','valHist','totalEpochs','ns','-v7.3');
fprintf('    [cp] %s (val=%.4f)\n',f,bestVal);
end
