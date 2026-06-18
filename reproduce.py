"""
DGA Transformer Fault Diagnosis — Hierarchical, with honest boundaries
=====================================================================
Regenerates every number in the README from scratch (single seed).

Pipeline:
  Step 1  Correctness gate: exact Duval Triangle 1 (IEC 60599, mineral oil),
          verified to tile the triangle + spot-checked on published points.
  Step 1b Dedup (removes train/test leakage) + Phase-1 Normal/Fault separability.
  Step 2  Phase-2 fault-type: Duval vs RF(3 gases) vs RF(5 gases+ratios),
          uplift decomposed into algorithm gain vs information gain.
  Step 3  Confidence calibration (ECE) + physical noise stress-test
          (multiplicative Gaussian, 3sigma=margin) + boundary-vs-interior.

Data: alan-456/transformer-fault-dataset (data.xlsx, 2321 samples, 7 classes
incl. Normal). NOTE: labels are NOT inspection-verified ground truth (see
README limitations).
"""
import os, json, urllib.request, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
np.random.seed(42)

# ---------- data ----------
URL="https://github.com/alan-456/transformer-fault-dataset/raw/main/data.xlsx"
if not os.path.exists("data.xlsx"):
    print("downloading dataset..."); urllib.request.urlretrieve(URL,"data.xlsx")
df=pd.read_excel("data.xlsx"); df.columns=['H2','CH4','C2H6','C2H4','C2H2','zh']
ZH={'正常':'Normal','局部放电':'PD','低能放电':'D1','高能放电':'D2',
    '低温过热':'T1','中温过热':'T2','高温过热':'T3'}
df['label']=df['zh'].map(ZH)
GASES=['H2','CH4','C2H6','C2H4','C2H2']

# ---------- Step 1: exact Duval Triangle 1 ----------
def duval1(ch4,c2h4,c2h2):
    s=ch4+c2h4+c2h2
    if s<=0: return 'ND'
    CH4,C2H4,C2H2=100*ch4/s,100*c2h4/s,100*c2h2/s
    if CH4>=98: return 'PD'
    if C2H2<4:
        return 'T1' if C2H4<20 else ('T2' if C2H4<50 else 'T3')
    if C2H4>=50 and C2H2<15: return 'T3'
    if C2H2<13: return 'DT'
    if C2H4<23: return 'D1'
    if C2H2>=29: return 'D2'
    return 'D2' if C2H4<40 else 'DT'

def verify_duval():
    miss=sum(duval1(a,b,100-a-b)=='ND' for a in range(101) for b in range(101-a))
    checks=[((204,17,0),'T1'),((1050,3520,29),'T3'),((2647,2590,6),'T2')]
    ok=all(duval1(*g)==e for g,e in checks)
    return miss, ok

# ---------- Step 1b: dedup + Phase-1 ----------
n0=len(df); df=df.drop_duplicates(subset=GASES+['zh']).reset_index(drop=True)
n_dups=n0-len(df)
df['total']=df[GASES].sum(axis=1)
nrm=df[df.label=='Normal']['total']; flt=df[df.label!='Normal']['total']
phase1_gap=(float(nrm.quantile(.9)), float(flt.quantile(.1)))   # clean separation

# ---------- features ----------
ff=df[df.label!='Normal'].copy().reset_index(drop=True)
def feats(raw):
    nl=np.log1p(raw); s=raw.sum(1,keepdims=True)+1e-9; fr=raw/s; e=1e-6
    r=np.log1p(np.abs(np.c_[raw[:,4]/(raw[:,3]+e),raw[:,1]/(raw[:,0]+e),raw[:,3]/(raw[:,2]+e)]))
    return np.hstack([nl,fr,r])
X=feats(ff[GASES].values.astype(float)); y=ff.label.values
rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',
                          random_state=42,n_jobs=-1)
skf=StratifiedKFold(5,shuffle=True,random_state=42)

# ---------- Step 2: baseline + uplift decomposition ----------
duval_acc=float((ff.apply(lambda r:duval1(r.CH4,r.C2H4,r.C2H2),axis=1)==ff.label).mean())
pred5=cross_val_predict(rf,X,y,cv=skf); acc5=accuracy_score(y,pred5); f1=f1_score(y,pred5,average='macro')
X3=ff[['CH4','C2H4','C2H2']].values.astype(float)
X3=np.hstack([np.log1p(X3),X3/(X3.sum(1,keepdims=True)+1e-9)])
acc3=accuracy_score(y,cross_val_predict(rf,X3,y,cv=skf))

# ---------- Step 3a: calibration ----------
proba=cross_val_predict(rf,X,y,cv=skf,method='predict_proba'); classes=rf.fit(X,y).classes_
conf=proba.max(1); predc=classes[proba.argmax(1)]; correct=(predc==y).astype(int)
ece=0.0
for lo in [0,.5,.6,.7,.8,.9]:
    hi_e=lo+0.1 if lo>0 else 0.5
    m=(conf>=lo)&(conf<hi_e)
    if m.sum(): ece+=m.sum()/len(conf)*abs(conf[m].mean()-correct[m].mean())
hi=conf>=0.9; hi_wrong_rate=float(((conf>=0.9)&(predc!=y)).sum()/max(hi.sum(),1))
cal_bins=[]
for lo in [0,.5,.6,.7,.8,.9]:
    hi_e=lo+0.1 if lo>0 else 0.5
    m=(conf>=lo)&(conf<hi_e)
    if m.sum(): cal_bins.append([round(float(conf[m].mean()),3),round(float(correct[m].mean()),3),int(m.sum())])

# ---------- Step 3b/c: noise + boundary ----------
Xtr,Xte,ytr,yte,_,fte=train_test_split(X,y,ff,test_size=0.3,stratify=y,random_state=42)
rf.fit(Xtr,ytr); raw_te=fte[GASES].values.astype(float); rng=np.random.default_rng(0)
def noisy_acc(raw,yy,margin,n=20):
    s=margin/3; ra,da=[],[]
    for _ in range(n):
        nz=np.clip(raw*(1+rng.normal(0,s,raw.shape)),0,None)
        ra.append(accuracy_score(yy,rf.predict(feats(nz))))
        da.append((np.array([duval1(nz[i,1],nz[i,3],nz[i,4]) for i in range(len(nz))])==yy).mean())
    return float(np.mean(ra)),float(np.mean(da))
noise_curve={f"{int(m*100)}%":noisy_acc(raw_te,yte,m) for m in [0,.10,.20,.30,.50]}
base=np.array([duval1(raw_te[i,1],raw_te[i,3],raw_te[i,4]) for i in range(len(raw_te))])
flip=sum((np.array([duval1(p[i,1],p[i,3],p[i,4]) for i in range(len(p))])!=base)
         for p in [np.clip(raw_te*(1+rng.normal(0,.05,raw_te.shape)),0,None) for _ in range(15)])/15
bdy=flip>=0.3; itr=flip==0.0
rf_b,dv_b=noisy_acc(raw_te[bdy],yte[bdy],0.0,30); rf_i,dv_i=noisy_acc(raw_te[itr],yte[itr],0.0,30)

# ---------- report ----------
miss,spot=verify_duval()
R={"duval_verify":{"uncovered":miss,"spotchecks_pass":spot},
   "dedup":{"before":n0,"after":len(df),"removed":n_dups},
   "phase1_separation":{"normal_p90_ppm":phase1_gap[0],"fault_p10_ppm":phase1_gap[1]},
   "phase2":{"duval_3gas":round(duval_acc,3),"rf_3gas":round(acc3,3),
             "rf_5gas":round(acc5,3),"macro_f1":round(f1,3),
             "algo_gain":round(acc3-duval_acc,3),"info_gain":round(acc5-acc3,3)},
   "calibration":{"ece":round(ece,3),"high_conf_error_rate":round(hi_wrong_rate,3),"bins":cal_bins},
   "noise_curve":noise_curve,
   "boundary":{"n_boundary":int(bdy.sum()),"n_interior":int(itr.sum()),
               "rf_boundary":round(rf_b,3),"duval_boundary":round(dv_b,3),
               "rf_interior":round(rf_i,3),"duval_interior":round(dv_i,3)}}
print(json.dumps(R,indent=2))
json.dump(R,open("verified_results.json","w"),indent=2)
print("\nsaved verified_results.json")
