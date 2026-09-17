import os
# Reproducibility controls should be configured before importing numerical libraries.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ============================================================
# 0. Settings
# ============================================================
# Fixed, pre-specified seeds. Do not tune this list against test performance.
BASE_SEED = 42
SEEDS = (42, 0, 1, 7, 123, 17, 99, 2024, 314, 999)
N_RUNS = len(SEEDS)

def set_global_seed(seed: int) -> None:
    """Set all RNGs used by this script to a known state."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Force deterministic CPU/GPU kernels whenever PyTorch provides them.
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

set_global_seed(BASE_SEED)

STEADY_SHIFT_STEPS = int(os.environ.get('STEADY_SHIFT_STEPS', '5'))
GLOBAL_EPOCHS      = 100
GLOBAL_LR          = 0.002
GLOBAL_WD          = 5e-4
GLOBAL_BATCH       = 32
R = 3
sL = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. Load Data
# ============================================================
print("=== 1. Loading Data ===")

def load_csv_groups(file_names):
    all_data = []; groups = []; start = 0
    for fn in file_names:
        path = os.path.join(BASE_DIR, fn)
        data = np.loadtxt(path, delimiter=',', encoding='utf-8-sig')
        data = np.asarray(data, dtype=float)
        data[:, 3] = np.maximum(data[:, 3], 0.0)
        data[:, :3] = data[:, :3] - data[0:1, :3]
        all_data.append(data)
        end = start + len(data) - 1
        groups.append((start, end))
        start = end + 1
    return np.vstack(all_data), groups

train_files = ['train1.csv']
test_files  = ['test1.csv']
train_raw, grp_tr1 = load_csv_groups(train_files)
test_raw,  grp_te1 = load_csv_groups(test_files)
T_tr1 = train_raw[:, :3]; E_tr1 = train_raw[:, 3]
T_te1 = test_raw[:, :3];  E_te1 = test_raw[:, 3]

# ============================================================
# 2. Compute Lags
# ============================================================
print("=== 2. Computing Lags ===")

def matern_52_kernel(X1, X2, h):
    dist = np.abs(X1[:, None] - X2[None, :])
    r = np.sqrt(5.0) * dist / (h + 1e-12)
    return (1.0 + r + 5.0*(dist**2)/(3.0*h**2+1e-12)) * np.exp(-r)

def krr_smooth(x, y, lam, h):
    n = len(x)
    idx_sub = np.linspace(0, n-1, min(n,2000), dtype=int) if n>2000 else np.arange(n)
    xs = x[idx_sub]; ys = y[idx_sub]; ns = len(idx_sub)
    K_sub = matern_52_kernel(xs, xs, h)
    A = K_sub + ns*lam*np.eye(ns)
    try:    beta = np.linalg.solve(A, ys)
    except: beta = np.linalg.lstsq(A, ys, rcond=None)[0]
    return matern_52_kernel(x, xs, h) @ beta

def krr_gcv(x, y, lam_range=[1e-5,1e-4,1e-3,1e-2], h_range=None):
    n=len(x); x=np.asarray(x,float); y=np.asarray(y,float)
    xr=max(x.max()-x.min(),1e-10); xn=(x-x.min())/xr
    if h_range is None: h_range=np.array([50,100,200,500,1000,2000])
    h_range=np.asarray(h_range,float); hs=h_range/xr
    mg=min(n,500); ix=np.linspace(0,n-1,mg,dtype=int) if n>mg else np.arange(n)
    xsub=xn[ix]; ysub=y[ix]; ns=len(ix)
    bg=np.inf; lam_o=lam_range[0]; ho=h_range[0]
    K_all={h: matern_52_kernel(xsub,xsub,h) for h in hs if h>1e-12}
    for lam in lam_range:
        for hi_,h in enumerate(hs):
            if h<1e-12: continue
            K=K_all[h]; A=K+ns*lam*np.eye(ns)
            try:    invA=np.linalg.inv(A)
            except: invA=np.linalg.pinv(A)
            S=K@invA; yh=S@ysub; tv=np.trace(S)
            rss=np.sum((ysub-yh)**2); den=ns*(1.0-tv/ns)**2
            gv=rss/den if den>1e-15 else np.inf
            if gv<bg: bg=gv; lam_o=lam; ho=h_range[hi_]
    return krr_smooth(xn, y, lam_o, ho/xr), lam_o, ho

def _residualize(v, Z=None):
    v=np.asarray(v,float).reshape(-1)
    vc=v-np.mean(v)
    if Z is None or Z.size==0:
        return vc
    Z=np.asarray(Z,float)
    if Z.ndim==1: Z=Z.reshape(-1,1)
    Zc=Z-np.mean(Z,axis=0,keepdims=True)
    if np.linalg.matrix_rank(Zc) == 0:
        return vc
    coef=np.linalg.lstsq(Zc, vc, rcond=None)[0]
    return vc-Zc@coef

def _lagged_conditional_arrays(dE, dTs, r, lag, selected):
    # Align E(t) with T_r(t-lag). Already selected lagged temperature
    # variables are used as the conditioning set Z.
    n=len(dE)
    lag=int(lag)
    start=lag
    if selected:
        start=max(start, max(int(lg) for _,lg in selected))
    if start>=n-2:
        return None, None, None
    y=dE[start:n]
    x=dTs[start-lag:n-lag, r]
    if selected:
        Z=np.column_stack([dTs[start-int(lg):n-int(lg), rr] for rr,lg in selected])
    else:
        Z=None
    return y, x, Z

def _conditional_corr_beta_se(y, x, Z=None):
    # Conditional correlation: remove the linear influence of Z from both
    # E and T_r, then calculate the correlation between the two residuals.
    ry=_residualize(y, Z)
    rx=_residualize(x, Z)
    sy2=float(np.dot(ry,ry))
    sx2=float(np.dot(rx,rx))
    if sx2<1e-20 or sy2<1e-20:
        return 0.0, 0.0, np.inf
    cov=float(np.dot(ry,rx))
    alpha=cov/np.sqrt(sy2*sx2)
    beta=cov/(sx2+1e-12)
    se=float(np.mean((ry-beta*rx)**2))
    return alpha, beta, se

def _best_conditional_lag_for_feature(dE, dTs, r, selected, max_lag):
    n=len(dE)
    lag_max=min(int(max_lag), n-3)
    best_lag=0; best_alpha=0.0; best_beta=0.0; best_se=np.inf
    for lag in range(lag_max+1):
        y,x,Z=_lagged_conditional_arrays(dE,dTs,r,lag,selected)
        if y is None or len(y)<3:
            continue
        alpha,beta,se=_conditional_corr_beta_se(y,x,Z)
        # Main objective: maximize absolute conditional correlation strength.
        # Tie-breaker: choose the smaller conditional squared error, then the
        # shorter lag to avoid unstable long-lag choices when correlations are
        # numerically indistinguishable.
        if (abs(alpha)>abs(best_alpha)+1e-12 or
            (abs(abs(alpha)-abs(best_alpha))<=1e-12 and se<best_se-1e-12) or
            (abs(abs(alpha)-abs(best_alpha))<=1e-12 and abs(se-best_se)<=1e-12 and lag<best_lag)):
            best_lag=int(lag); best_alpha=alpha; best_beta=beta; best_se=se
    return best_lag, best_alpha, best_beta, best_se

def compute_lags(T, E, groups, max_lag=300):
    pgl=[]
    for gi,(gs,ge) in enumerate(groups):
        ng=ge-gs+1
        if ng<100: pgl.append([0]*R); continue
        x=np.arange(ng, dtype=float); Eg=E[gs:ge+1]
        hr=np.array([100,300,500,1000,2000])
        Es,_,_=krr_gcv(x,Eg,h_range=hr); dE=np.diff(Eg-Es)

        # WS residual differences for all temperature channels.
        dTs=[]
        for r in range(R):
            Tg=T[gs:ge+1,r]
            Ts,_,_=krr_gcv(x,Tg,h_range=hr)
            dTs.append(np.diff(Tg-Ts))
        dTs=np.column_stack(dTs)

        # Sequential conditional-correlation lag estimation.
        # No result from the original cross-correlation lag estimator is used as
        # a search constraint. Each candidate lag is selected by conditional
        # correlation, and each variable is added by the conditional SE criterion.
        lg=[0]*R
        selected=[]
        remaining=list(range(R))
        while remaining:
            cand=[]
            for r in remaining:
                if np.std(dTs[:,r])<1e-10 or np.std(dE)<1e-10:
                    cand.append((r,0,0.0,0.0,np.inf)); continue
                lag,alpha,beta,se=_best_conditional_lag_for_feature(dE,dTs,r,selected,max_lag)
                cand.append((r,lag,alpha,beta,se))

            # Following the document's sequential idea: after determining the
            # optimal lag for each remaining feature, choose the next feature by
            # the smallest conditional squared error.
            cand.sort(key=lambda z: (z[4], -abs(z[2]), z[0]))
            r,lag,alpha,beta,se=cand[0]
            lg[r]=lag
            selected.append((r,lag))
            remaining.remove(r)
        pgl.append(lg)
    return np.array(pgl)

tr_lags = compute_lags(T_tr1, E_tr1, grp_tr1)

# Print computed lag values to Terminal for checking/debugging.
# tr_lags: raw lag steps computed from the original sampling interval.
# tr_l60 : lag steps after converting to the downsampled scale, clipped to [1, 5].
print("\n--- Lag Calculation Results ---")
print("Raw train lags per group and channel, shape =", tr_lags.shape)
for gi, lags in enumerate(tr_lags, start=1):
    print(f"Group {gi}: T1_lag={lags[0]}, T2_lag={lags[1]}, T3_lag={lags[2]}")

tr_l60  = np.clip(np.round(np.median(tr_lags/60.0, axis=0)).astype(int), 1, 5)
te_l60  = tr_l60.copy()

print("Median raw train lags by channel:", np.median(tr_lags, axis=0))
print("Converted train lags on downsampled scale, tr_l60:", tr_l60)
print("Test lags copied from train lags, te_l60:", te_l60)
print("--------------------------------\n")

# ============================================================
# 3. Downsample + Feature Engineering
# ============================================================
print("=== 3. Preparing Data ===")

def downsample_ma(T, E, groups, win=5):
    hw=win//2; Td=[]; Ed=[]; gd=[]; idx=0
    for gs,ge in groups:
        ng=ge-gs+1; nw=max(1,ng//60); Ts=T[gs:ge+1]; Es=E[gs:ge+1]
        Tsm=np.zeros_like(Ts); Esm=np.zeros(len(Es))
        for i in range(len(Es)):
            lo=max(0,i-hw); hi_=min(len(Es),i+hw+1)
            Tsm[i]=Ts[lo:hi_].mean(axis=0); Esm[i]=Es[lo:hi_].mean()
        gs_d=idx
        for w in range(1,nw+1):
            pt=min(w*60,ng)-1; Td.append(Tsm[pt]); Ed.append(Esm[pt]); idx+=1
        gd.append((gs_d,idx-1))
    return np.array(Td), np.array(Ed), gd

T60_tr, E60_tr, g60_tr = downsample_ma(T_tr1, E_tr1, grp_tr1)
T60_te, E60_te, g60_te = downsample_ma(T_te1, E_te1, grp_te1)

def build_rate_feat(T, lags_60, groups):
    dT=np.zeros_like(T)
    for gs,ge in groups:
        for idx in range(gs,ge+1):
            for r in range(R):
                src=max(gs,idx-int(lags_60[r])); dT[idx,r]=T[idx,r]-T[src,r]
    return dT

dT60_tr = build_rate_feat(T60_tr, tr_l60, g60_tr)
dT60_te = build_rate_feat(T60_te, te_l60, g60_te)

T_std_tr  = T60_tr.std(axis=0) + 1e-8
dT_std_tr = dT60_tr.std(axis=0) + 1e-8
dT_scale  = T_std_tr / dT_std_tr

T60_tr_aug = np.hstack([T60_tr, dT60_tr * dT_scale])
T60_te_aug = np.hstack([T60_te, dT60_te * dT_scale])

# ============================================================
# 4. Phase Segmentation
# ============================================================
print("=== 4. Phase Segmentation ===")

def segment_phases(E, groups, shift_steps=0):
    ph=np.ones(len(E),dtype=int)
    for gs,ge in groups:
        ng=ge-gs+1; Eg=E[gs:ge+1]; Er=Eg.max()-Eg[0]
        if Er<3.0: tr=max(3,ng//6)
        else:
            th=Eg[0]+0.80*Er; tr=ng//3
            for j in range(3,ng-2):
                if Eg[j]>=th: tr=j; break
            tr=max(5,min(tr,ng-3))
        tr=max(5,min(tr+int(shift_steps),ng-3))
        ph[gs:gs+tr]=0
    return ph

ph_tr = segment_phases(E60_tr, g60_tr, STEADY_SHIFT_STEPS)
train_heat_ratio = np.mean(ph_tr == 0)

ph_te = np.ones(len(E60_te), dtype=int)
for gs, ge in g60_te:
    ng = ge - gs + 1
    tr = max(5, int(ng * train_heat_ratio))
    ph_te[gs:gs+tr] = 0

# ============================================================
# 5. Build Sequences
# ============================================================
print("=== 5. Building Model Data ===")

def build_seqs_sc(T, E, groups, sL):
    X,Y=[],[]
    for gs,ge in groups:
        for idx in range(gs,ge+1):
            ws=max(gs,idx-sL+1); seq=T[ws:idx+1]
            if len(seq)<sL: seq=np.vstack([np.tile(seq[0:1],(sL-len(seq),1)),seq])
            X.append(seq); Y.append(E[idx])
    return np.array(X), np.array(Y)

def build_seqs_mc(T, dT, E, groups, sL, dT_scale):
    X,Y=[],[]
    for gs,ge in groups:
        for idx in range(gs,ge+1):
            ws=max(gs,idx-sL+1)
            seq_T  = T[ws:idx+1]
            seq_dT = dT[ws:idx+1] * dT_scale
            if len(seq_T)<sL:
                pad=sL-len(seq_T)
                seq_T  = np.vstack([np.tile(seq_T[0:1],(pad,1)),seq_T])
                seq_dT = np.vstack([np.tile(seq_dT[0:1],(pad,1)),seq_dT])
            X.append(np.stack([seq_T, seq_dT], axis=-1))
            Y.append(E[idx])
    return np.array(X), np.array(Y)

X_nl,      Y_nl      = build_seqs_sc(T60_tr,     E60_tr, g60_tr, sL)
X_gman_lg, Y_gman_lg = build_seqs_mc(T60_tr, dT60_tr, E60_tr, g60_tr, sL, dT_scale)
X_lstm_lg, Y_lstm_lg = build_seqs_sc(T60_tr_aug, E60_tr, g60_tr, sL)

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
def tanh_act(x): return np.tanh(np.clip(x,-30,30))
def relu(x): return np.maximum(0,x)

# ============================================================
# 6. Models
# ============================================================
def train_rfr(X, Y, seed=42):
    mdl = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=1,
        max_features='sqrt', random_state=int(seed), n_jobs=1)
    mdl.fit(X, Y)
    return mdl

def predict_rfr(mdl, X): return mdl.predict(X)

def train_svr(X, Y, seed=42):
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(Y.reshape(-1,1))
    Xn = scaler_x.transform(X)
    Yn = scaler_y.transform(Y.reshape(-1,1)).ravel()
    param_grid = {
        'C':       [0.5, 1, 2],
        'degree':  [2, 3],
        'coef0':   [0.5, 1.0],
        'epsilon': [0.05, 0.1, 0.2]
    }
    grid = GridSearchCV(SVR(kernel='poly'), param_grid, cv=3,
                        scoring='neg_mean_squared_error', n_jobs=1)
    grid.fit(Xn, Yn)
    return {'svr': grid.best_estimator_, 'scaler_x': scaler_x, 'scaler_y': scaler_y}

def predict_svr(mdl, X):
    Xn = mdl['scaler_x'].transform(X)
    Yn = mdl['svr'].predict(Xn)
    return mdl['scaler_y'].inverse_transform(Yn.reshape(-1,1)).ravel()

class GMANSingleChannel(nn.Module):
    def __init__(self, nF=3, sL=3, dH=16, dropout=0.2):
        super().__init__()
        self.dH=dH
        self.input_proj=nn.Linear(1, dH)
        self.SE=nn.Parameter(torch.randn(nF,dH)*0.1)
        self.TE=nn.Parameter(torch.randn(sL,dH)*0.1)
        self.WQs=nn.Linear(dH,dH,bias=False); self.WKs=nn.Linear(dH,dH,bias=False); self.WVs=nn.Linear(dH,dH,bias=False)
        self.WQt=nn.Linear(dH,dH,bias=False); self.WKt=nn.Linear(dH,dH,bias=False); self.WVt=nn.Linear(dH,dH,bias=False)
        self.gate=nn.Linear(dH, dH)
        self.drop=nn.Dropout(dropout)
        self.output_fc=nn.Sequential(nn.Linear(dH,dH), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dH,1))
    def forward(self, x):
        B,T,N=x.shape; dk=self.dH**0.5
        H=self.input_proj(x.unsqueeze(-1)) + self.SE.unsqueeze(0).unsqueeze(0) + self.TE.unsqueeze(0).unsqueeze(2)
        HS=torch.stack([
            torch.softmax(self.WQs(H[:,t])@self.WKs(H[:,t]).transpose(-2,-1)/dk, dim=-1)@self.WVs(H[:,t])
            for t in range(T)], dim=1)
        HT=torch.stack([
            torch.softmax(self.WQt(H[:,:,n])@self.WKt(H[:,:,n]).transpose(-2,-1)/dk, dim=-1)@self.WVt(H[:,:,n])
            for n in range(N)], dim=2)
        g=torch.sigmoid(self.gate(HS+HT))
        return self.output_fc(self.drop((g*HS+(1-g)*HT).mean(dim=(1,2)))).squeeze(-1)

class GMANMultiChannel(nn.Module):
    def __init__(self, nF=3, nC=2, sL=3, dH=16, dropout=0.2):
        super().__init__()
        self.dH=dH
        self.input_proj=nn.Linear(nC, dH)
        self.SE=nn.Parameter(torch.randn(nF,dH)*0.1)
        self.TE=nn.Parameter(torch.randn(sL,dH)*0.1)
        self.WQs=nn.Linear(dH,dH,bias=False); self.WKs=nn.Linear(dH,dH,bias=False); self.WVs=nn.Linear(dH,dH,bias=False)
        self.WQt=nn.Linear(dH,dH,bias=False); self.WKt=nn.Linear(dH,dH,bias=False); self.WVt=nn.Linear(dH,dH,bias=False)
        self.gate=nn.Linear(dH, dH)
        self.drop=nn.Dropout(dropout)
        self.output_fc=nn.Sequential(nn.Linear(dH,dH), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dH,1))
    def forward(self, x):
        B,T,N,nC=x.shape; dk=self.dH**0.5
        H=self.input_proj(x) + self.SE.unsqueeze(0).unsqueeze(0) + self.TE.unsqueeze(0).unsqueeze(2)
        HS=torch.stack([
            torch.softmax(self.WQs(H[:,t])@self.WKs(H[:,t]).transpose(-2,-1)/dk, dim=-1)@self.WVs(H[:,t])
            for t in range(T)], dim=1)
        HT=torch.stack([
            torch.softmax(self.WQt(H[:,:,n])@self.WKt(H[:,:,n]).transpose(-2,-1)/dk, dim=-1)@self.WVt(H[:,:,n])
            for n in range(N)], dim=2)
        g=torch.sigmoid(self.gate(HS+HT))
        return self.output_fc(self.drop((g*HS+(1-g)*HT).mean(dim=(1,2)))).squeeze(-1)

def _train_gman_once(model, X, Y, epochs, lr, wd, seed, is_mc):
    set_global_seed(seed)
    if is_mc:
        Xm=X.mean(axis=(0,1,2)); Xsd=X.std(axis=(0,1,2))+1e-8
    else:
        Xm=X.mean(axis=(0,1)); Xsd=X.std(axis=(0,1))+1e-8
    Xn=(X-Xm)/Xsd
    Ym=Y.mean(); Ysd=Y.std()+1e-8; Yn=(Y-Ym)/Ysd
    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loader=torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(Xn,dtype=torch.float32),
            torch.tensor(Yn,dtype=torch.float32)),
        batch_size=GLOBAL_BATCH, shuffle=True)
    model.train()
    for ep in range(epochs):
        for xb,yb in loader:
            optimizer.zero_grad(); loss=nn.MSELoss()(model(xb),yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        yp=model(torch.tensor(Xn,dtype=torch.float32)).numpy()
    train_mse=float(np.mean((yp-Yn)**2))
    return {'model':model,'Xm':Xm,'Xsd':Xsd,'Ym':Ym,'Ysd':Ysd,'is_mc':is_mc}, train_mse

def train_gman_nl(X, Y, dH=16, dropout=0.2):
    best_mdl=None; best_mse=np.inf
    for seed in SEEDS:
        set_global_seed(seed)  # must precede parameter initialization
        model=GMANSingleChannel(nF=X.shape[2], sL=X.shape[1], dH=dH, dropout=dropout)
        mdl, mse = _train_gman_once(model, X, Y, GLOBAL_EPOCHS, GLOBAL_LR, GLOBAL_WD, seed, is_mc=False)
        if mse < best_mse: best_mse=mse; best_mdl=mdl
    return best_mdl

def train_gman_lg(X, Y, dH=16, dropout=0.2):
    best_mdl=None; best_mse=np.inf
    for seed in SEEDS:
        set_global_seed(seed)  # must precede parameter initialization
        model=GMANMultiChannel(nF=X.shape[2], nC=X.shape[3], sL=X.shape[1], dH=dH, dropout=dropout)
        mdl, mse = _train_gman_once(model, X, Y, GLOBAL_EPOCHS, GLOBAL_LR, GLOBAL_WD, seed, is_mc=True)
        if mse < best_mse: best_mse=mse; best_mdl=mdl
    return best_mdl

def predict_gman_batch(mdl, seqs):
    Xn=(seqs-mdl['Xm'])/mdl['Xsd']
    with torch.no_grad(): yp=mdl['model'](torch.tensor(Xn,dtype=torch.float32)).numpy()
    return yp*mdl['Ysd']+mdl['Ym']

def _train_lstm_once(X, Y, nH, epochs, lr, wd, seed):
    set_global_seed(seed)
    N,sL_t,nF=X.shape
    Xm=X.mean(axis=(0,1)); Xsd=X.std(axis=(0,1))+1e-8; Xn=(X-Xm)/Xsd
    Ym=Y.mean(); Ysd=Y.std()+1e-8; Yn=(Y-Ym)/Ysd
    sc=np.sqrt(2.0/(nF+nH))
    Wf=np.random.randn(nH,nF+nH)*sc; bf=np.ones(nH)
    Wi=np.random.randn(nH,nF+nH)*sc; bi=np.zeros(nH)
    Wc=np.random.randn(nH,nF+nH)*sc; bc=np.zeros(nH)
    Wo=np.random.randn(nH,nF+nH)*sc; bo=np.zeros(nH)
    Wa=np.random.randn(1,nH)*sc; ba=np.zeros(1)
    Wfc=np.random.randn(1,nH)*sc; bfc=np.zeros(1)
    plist=[Wf,Wi,Wc,Wo,bf,bi,bc,bo,Wa,ba,Wfc,bfc]
    beta1=0.9; beta2=0.999; eps=1e-8
    m_adam=[np.zeros_like(p) for p in plist]
    v_adam=[np.zeros_like(p) for p in plist]
    t_adam=0; batch_size=GLOBAL_BATCH
    for ep in range(epochs):
        cur_lr=lr*(0.5**(ep//50))
        perm=np.random.permutation(N)
        grad_acc=[np.zeros_like(p) for p in plist]
        for ni in range(N):
            idx=perm[ni]; xs=Xn[idx]; yt=Yn[idx]
            hp=np.zeros(nH); cp=np.zeros(nH)
            hs=[]; cs=[]; fl=[]; il=[]; gl=[]; ol=[]
            for t in range(sL_t):
                z=np.concatenate([xs[t],hp])
                ft=sigmoid(Wf@z+bf); it=sigmoid(Wi@z+bi)
                gt=tanh_act(Wc@z+bc); ot=sigmoid(Wo@z+bo)
                ct=ft*cp+it*gt; ht=ot*tanh_act(ct)
                hs.append(ht); cs.append(ct); fl.append(ft); il.append(it); gl.append(gt); ol.append(ot)
                hp=ht; cp=ct
            H=np.array(hs); sc2=(H@Wa.T+ba).flatten()
            al=np.exp(sc2-sc2.max()); al/=al.sum()+1e-8
            ctx=al@H; yp=(Wfc@ctx+bfc)[0]; dL=yp-yt
            dWfc=dL*ctx.reshape(1,-1); dbfc=np.array([dL]); dctx=dL*Wfc.flatten()
            dH_arr=np.zeros_like(H)
            for t in range(sL_t): dH_arr[t]+=al[t]*dctx
            da=H@dctx; ds=al*(da-np.dot(al,da))
            dWa=ds.reshape(-1,1).T@H; dba=ds.sum(keepdims=True)
            for t in range(sL_t): dH_arr[t]+=ds[t]*Wa.flatten()
            dWfa=np.zeros_like(Wf); dWia=np.zeros_like(Wi)
            dWca=np.zeros_like(Wc); dWoa=np.zeros_like(Wo)
            dbfa=np.zeros_like(bf); dbia=np.zeros_like(bi)
            dbca=np.zeros_like(bc); dboa=np.zeros_like(bo)
            dhn=np.zeros(nH); dcn=np.zeros(nH)
            for t in reversed(range(sL_t)):
                dh=dH_arr[t]+dhn; dot2=dh*tanh_act(cs[t])
                dct=dh*ol[t]*(1-tanh_act(cs[t])**2)+dcn
                dft=dct*(cs[t-1] if t>0 else np.zeros(nH))
                dit=dct*gl[t]; dgt=dct*il[t]
                dfg=dft*fl[t]*(1-fl[t]); dig=dit*il[t]*(1-il[t])
                dgg=dgt*(1-gl[t]**2); dog=dot2*ol[t]*(1-ol[t])
                htm=hs[t-1] if t>0 else np.zeros(nH)
                z=np.concatenate([xs[t],htm])
                dWfa+=np.outer(dfg,z); dWia+=np.outer(dig,z)
                dWca+=np.outer(dgg,z); dWoa+=np.outer(dog,z)
                dbfa+=dfg; dbia+=dig; dbca+=dgg; dboa+=dog
                dz=Wf.T@dfg+Wi.T@dig+Wc.T@dgg+Wo.T@dog
                dhn=dz[nF:]; dcn=dct*fl[t]
            grads=[dWfa,dWia,dWca,dWoa,dbfa,dbia,dbca,dboa,dWa,dba,dWfc,dbfc]
            for gi in range(len(grads)): grad_acc[gi]+=grads[gi]
            if (ni+1)%batch_size==0 or ni==N-1:
                t_adam+=1
                bs=batch_size if (ni+1)%batch_size==0 else ((ni+1)%batch_size)
                for gi in range(len(plist)):
                    g=grad_acc[gi]/bs
                    if plist[gi].ndim>1: g+=wd*plist[gi]
                    gnorm=np.linalg.norm(g)
                    if gnorm>5.0: g=g*5.0/gnorm
                    m_adam[gi]=beta1*m_adam[gi]+(1-beta1)*g
                    v_adam[gi]=beta2*v_adam[gi]+(1-beta2)*g**2
                    mh=m_adam[gi]/(1-beta1**t_adam)
                    vh=v_adam[gi]/(1-beta2**t_adam)
                    plist[gi]-=cur_lr*mh/(np.sqrt(vh)+eps)
                grad_acc=[np.zeros_like(p) for p in plist]
        Wf,Wi,Wc,Wo,bf,bi,bc,bo,Wa,ba,Wfc,bfc=plist
    train_errs=[]
    for i in range(N):
        xs=Xn[i]; hp=np.zeros(nH); cp=np.zeros(nH); hs2=[]
        for t in range(sL_t):
            z=np.concatenate([xs[t],hp])
            ft=sigmoid(Wf@z+bf); it=sigmoid(Wi@z+bi)
            gt=tanh_act(Wc@z+bc); ot=sigmoid(Wo@z+bo)
            ct=ft*cp+it*gt; ht=ot*tanh_act(ct)
            hs2.append(ht); hp=ht; cp=ct
        H2=np.array(hs2); s2=(H2@Wa.T+ba).flatten()
        al2=np.exp(s2-s2.max()); al2/=al2.sum()+1e-8
        yp2=(Wfc@(al2@H2)+bfc)[0]
        train_errs.append((yp2-Yn[i])**2)
    train_mse=float(np.mean(train_errs))
    return {'Wf':Wf,'Wi':Wi,'Wc':Wc,'Wo':Wo,'bf':bf,'bi':bi,'bc':bc,'bo':bo,
            'Wa':Wa,'ba':ba,'Wfc':Wfc,'bfc':bfc,
            'Xm':Xm,'Xsd':Xsd,'Ym':Ym,'Ysd':Ysd,'nH':nH}, train_mse

def train_lstm(X, Y, nH=24):
    best_mdl=None; best_mse=np.inf
    for seed in SEEDS:
        mdl, mse = _train_lstm_once(X, Y, nH, GLOBAL_EPOCHS, GLOBAL_LR, GLOBAL_WD, seed)
        if mse < best_mse: best_mse=mse; best_mdl=mdl
    return best_mdl

def predict_lstm(mdl, seq):
    nH=mdl['nH']; Xn=(seq-mdl['Xm'])/mdl['Xsd']
    h=np.zeros(nH); c=np.zeros(nH); hs=[]
    for t in range(len(Xn)):
        z=np.concatenate([Xn[t],h])
        ft=sigmoid(mdl['Wf']@z+mdl['bf']); it=sigmoid(mdl['Wi']@z+mdl['bi'])
        gt=tanh_act(mdl['Wc']@z+mdl['bc']); ot=sigmoid(mdl['Wo']@z+mdl['bo'])
        c=ft*c+it*gt; h=ot*tanh_act(c); hs.append(h)
    H=np.array(hs); s=(H@mdl['Wa'].T+mdl['ba']).flatten()
    al=np.exp(s-s.max()); al/=al.sum()+1e-8
    yn=(mdl['Wfc']@(al@H)+mdl['bfc'])[0]
    return yn*mdl['Ysd']+mdl['Ym']

# Empirical TSDANN settings for short records (raw length about 1380, three input channels).
# Lambda is non-zero and gradually increased, so the domain-adversarial branch is active
# while avoiding unstable early updates on the small downsampled sample size.
TSDANN_LAMBDA_MAX   = 0.20
TSDANN_DISC_LR_MULT = 0.50
TSDANN_SELECT_ALPHA = 1.0e-2

def _train_tsdann_once(Xs, Ys, Xt, nH_fe, nH_pred, epochs, lr, lam, wd, seed):
    set_global_seed(seed)
    Ns=len(Ys); Nt=len(Xt); nF=Xs.shape[1]
    if Nt <= 0:
        raise ValueError('Xt must contain unlabeled target-domain inputs for TSDANN.')

    # Source-only normalization and source-only target scaling.
    # No target-domain labels are accepted by this function or used below.
    Ym=Ys.mean(); Ysd=Ys.std()+1e-8; Yn=(Ys-Ym)/Ysd
    Xm=Xs.mean(axis=0); Xsd=Xs.std(axis=0)+1e-8
    Xsn=(Xs-Xm)/Xsd; Xtn=(Xt-Xm)/Xsd

    sc_fe=np.sqrt(2.0/(nF+nH_fe))
    sc_pr=np.sqrt(2.0/(nH_fe+nH_pred))
    Wfe=np.random.randn(nH_fe,nF)*sc_fe; bfe=np.zeros(nH_fe)
    Wp=np.random.randn(nH_pred,nH_fe)*sc_pr; bp=np.zeros(nH_pred)
    Wo_p=np.random.randn(1,nH_pred)*sc_pr; bo_p=np.zeros(1)
    Wd=np.random.randn(1,nH_fe)*0.05; bd=np.zeros(1)

    total_steps=max(1, epochs*Ns)
    step_id=0
    for ep in range(epochs):
        perm_s=np.random.permutation(Ns); perm_t=np.random.permutation(Nt)
        for ni in range(Ns):
            step_id += 1
            p=step_id/total_steps
            lam_now=lam*(2.0/(1.0+np.exp(-10.0*p))-1.0)

            xs=Xsn[perm_s[ni]]; yt=Yn[perm_s[ni]]; xt=Xtn[perm_t[ni%Nt]]

            # Forward pass: source regression + binary domain discrimination.
            zs=Wfe@xs+bfe; fs=relu(zs)
            hp=relu(Wp@fs+bp); yp=(Wo_p@hp+bo_p)[0]
            zt=Wfe@xt+bfe; ft=relu(zt)
            ds=sigmoid(Wd@fs+bd)[0]  # source domain label = 1
            dt=sigmoid(Wd@ft+bd)[0]  # target domain label = 0

            # Supervised source-regression gradients.
            dL=yp-yt
            dWo_p=dL*hp.reshape(1,-1); dbo_p=np.array([dL])
            dhp=dL*Wo_p.flatten()*(hp>0)
            dWp=np.outer(dhp,fs); dbp=dhp
            dfs_pred=Wp.T@dhp*(zs>0)

            # Domain-classifier gradients for BCE loss:
            # Ld = -log(D(fs)) - log(1-D(ft)).
            # The discriminator minimizes Ld; the feature extractor receives
            # the reversed gradient, scaled by lam_now.
            gd_s=ds-1.0
            gd_t=dt
            dWd=gd_s*fs.reshape(1,-1) + gd_t*ft.reshape(1,-1)
            dbd=np.array([gd_s+gd_t])
            dfs_adv=(-lam_now)*(Wd.flatten()*gd_s)*(zs>0)
            dft_adv=(-lam_now)*(Wd.flatten()*gd_t)*(zt>0)

            dWfe_s=np.outer(dfs_pred+dfs_adv, xs); dbfe_s=dfs_pred+dfs_adv
            dWfe_t=np.outer(dft_adv, xt);          dbfe_t=dft_adv

            for g in [dWo_p,dWp,dWfe_s,dWfe_t,dWd]: np.clip(g,-1,1,out=g)
            np.clip(dbfe_s,-1,1,out=dbfe_s); np.clip(dbfe_t,-1,1,out=dbfe_t)
            np.clip(dbp,-1,1,out=dbp); np.clip(dbo_p,-1,1,out=dbo_p); np.clip(dbd,-1,1,out=dbd)

            dWo_p+=wd*Wo_p; dWp+=wd*Wp; dWfe_s+=wd*Wfe; dWfe_t+=wd*Wfe; dWd+=wd*Wd
            Wo_p-=lr*dWo_p; bo_p-=lr*dbo_p
            Wp-=lr*dWp; bp-=lr*dbp
            Wfe-=lr*(dWfe_s+dWfe_t); bfe-=lr*(dbfe_s+dbfe_t)
            Wd-=lr*TSDANN_DISC_LR_MULT*dWd; bd-=lr*TSDANN_DISC_LR_MULT*dbd

    train_errs=[]
    for i in range(Ns):
        fs2=relu(Wfe@Xsn[i]+bfe); hp2=relu(Wp@fs2+bp)
        yp2=(Wo_p@hp2+bo_p)[0]
        train_errs.append((yp2-Yn[i])**2)
    train_mse=float(np.mean(train_errs))

    ds_all=sigmoid((Xsn@Wfe.T+bfe).clip(min=0)@Wd.T+bd).ravel()
    dt_all=sigmoid((Xtn@Wfe.T+bfe).clip(min=0)@Wd.T+bd).ravel()
    domain_acc=0.5*(np.mean(ds_all>=0.5)+np.mean(dt_all<0.5))
    select_score=train_mse + TSDANN_SELECT_ALPHA*abs(domain_acc-0.5)

    return {'Wfe':Wfe,'bfe':bfe,'Wp':Wp,'bp':bp,'Wo':Wo_p,'bo':bo_p,
            'Xm':Xm,'Xsd':Xsd,'Ym':Ym,'Ysd':Ysd,
            'lam':lam,'domain_acc':float(domain_acc)}, train_mse, select_score

def train_tsdann(Xs, Ys, Xt, nH_fe=32, nH_pred=16, lam=TSDANN_LAMBDA_MAX):
    best_mdl=None; best_score=np.inf
    for seed in SEEDS:
        mdl, mse, score = _train_tsdann_once(Xs, Ys, Xt, nH_fe, nH_pred,
                                             GLOBAL_EPOCHS, GLOBAL_LR, lam, GLOBAL_WD, seed)
        if score < best_score: best_score=score; best_mdl=mdl
    return best_mdl

def predict_tsdann(mdl, x):
    xn=(x-mdl['Xm'])/mdl['Xsd']
    fs=relu(mdl['Wfe']@xn+mdl['bfe'])
    hp=relu(mdl['Wp']@fs+mdl['bp'])
    return (mdl['Wo']@hp+mdl['bo'])[0]*mdl['Ysd']+mdl['Ym']

# ============================================================
# 7. Train Models
# ============================================================
print("=== 7. Training Models ===")
print("Reproducibility protocol:", {
    "base_seed": BASE_SEED,
    "seeds": SEEDS,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
})

m_rfr_nl = train_rfr(T60_tr, E60_tr)
m_rfr_lg = train_rfr(T60_tr_aug, E60_tr)

m_svr_nl = train_svr(T60_tr, E60_tr)
m_svr_lg = train_svr(T60_tr_aug, E60_tr)

m_gman_nl = train_gman_nl(X_nl, Y_nl, dH=16)
m_gman_lg = train_gman_lg(X_gman_lg, Y_gman_lg, dH=16)

m_lstm_nl = train_lstm(X_nl, Y_nl, nH=24)
m_lstm_lg = train_lstm(X_lstm_lg, Y_lstm_lg, nH=24)

m_tsdann_nl = train_tsdann(T60_tr, E60_tr, T60_te, nH_fe=32, nH_pred=16)
m_tsdann_lg = train_tsdann(T60_tr_aug, E60_tr, T60_te_aug, nH_fe=32, nH_pred=16)

# ============================================================
# 8. Prediction & Evaluation
# ============================================================
print("=== 8. Prediction & Evaluation ===")

pred_rfr_nl_all  = predict_rfr(m_rfr_nl,  T60_te)
pred_rfr_lg_all  = predict_rfr(m_rfr_lg,  T60_te_aug)
pred_svr_nl_all  = predict_svr(m_svr_nl,  T60_te)
pred_svr_lg_all  = predict_svr(m_svr_lg,  T60_te_aug)

def build_test_seqs_sc(T, groups, sL):
    seqs=[]
    for i in range(len(T)):
        gi=0
        for g,(gs,ge) in enumerate(groups):
            if gs<=i<=ge: gi=g; break
        gsg=groups[gi][0]; ws=max(gsg,i-sL+1); seq=T[ws:i+1]
        if len(seq)<sL: seq=np.vstack([np.tile(seq[0:1],(sL-len(seq),1)),seq])
        seqs.append(seq)
    return np.array(seqs)

def build_test_seqs_mc(T, dT, groups, sL, dT_scale):
    seqs=[]
    for i in range(len(T)):
        gi=0
        for g,(gs,ge) in enumerate(groups):
            if gs<=i<=ge: gi=g; break
        gsg=groups[gi][0]; ws=max(gsg,i-sL+1)
        seq_T=T[ws:i+1]; seq_dT=dT[ws:i+1]*dT_scale
        if len(seq_T)<sL:
            pad=sL-len(seq_T)
            seq_T=np.vstack([np.tile(seq_T[0:1],(pad,1)),seq_T])
            seq_dT=np.vstack([np.tile(seq_dT[0:1],(pad,1)),seq_dT])
        seqs.append(np.stack([seq_T, seq_dT], axis=-1))
    return np.array(seqs)

test_seqs_nl_all  = build_test_seqs_sc(T60_te, g60_te, sL)
test_seqs_gman_lg = build_test_seqs_mc(T60_te, dT60_te, g60_te, sL, dT_scale)

pred_gman_nl_all = predict_gman_batch(m_gman_nl, test_seqs_nl_all)
pred_gman_lg_all = predict_gman_batch(m_gman_lg, test_seqs_gman_lg)

nTest = len(E60_te)
pred_lstm_nl_all   = np.zeros(nTest)
pred_lstm_lg_all   = np.zeros(nTest)
pred_tsdann_nl_all = np.zeros(nTest)
pred_tsdann_lg_all = np.zeros(nTest)

for i in range(nTest):
    gi=0
    for g,(gs,ge) in enumerate(g60_te):
        if gs<=i<=ge: gi=g; break
    gsg=g60_te[gi][0]
    ws=max(gsg,i-sL+1); snl=T60_te[ws:i+1]
    if len(snl)<sL: snl=np.vstack([np.tile(snl[0:1],(sL-len(snl),1)),snl])
    saug=T60_te_aug[ws:i+1]
    if len(saug)<sL: saug=np.vstack([np.tile(saug[0:1],(sL-len(saug),1)),saug])
    pred_lstm_nl_all[i]   = predict_lstm(m_lstm_nl, snl)
    pred_lstm_lg_all[i]   = predict_lstm(m_lstm_lg, saug)
    pred_tsdann_nl_all[i] = predict_tsdann(m_tsdann_nl, T60_te[i])
    pred_tsdann_lg_all[i] = predict_tsdann(m_tsdann_lg, T60_te_aug[i])

def compute_rmse_mae(y_true, y_pred):
    err=np.asarray(y_pred,float)-np.asarray(y_true,float)
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))

full_pred_all = [
    pred_rfr_nl_all,  pred_rfr_lg_all,
    pred_svr_nl_all,  pred_svr_lg_all,
    pred_gman_nl_all, pred_gman_lg_all,
    pred_lstm_nl_all, pred_lstm_lg_all,
    pred_tsdann_nl_all, pred_tsdann_lg_all
]
names_all = [
    'RFR_NoLag',    'RFR_Lag',
    'SVR_NoLag',    'SVR_Lag',
    'GMAN_NoLag',   'GMAN_Lag',
    'LSTM_NoLag',   'LSTM_Lag',
    'TSDANN_NoLag', 'TSDANN_Lag'
]

print("\nModel,RMSE,MAE")
for name, pred in zip(names_all, full_pred_all):
    rmse, mae = compute_rmse_mae(E60_te, pred)
    print(f"{name},{rmse:.6f},{mae:.6f}")
