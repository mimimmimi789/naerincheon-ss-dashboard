"""2단계: 사전학습 → 미세조정 후보를 운영 평가틀(fw: 2016–26 전진검증 + 홍수일 보정 ¾)과 잠금평가(≤2023 학습 → 2024–26)로 비교.
원천 자료는 각 학습 시점 이전 날짜만 사용(같은 날 같은 비 사상이 새지 않게)."""
import sys, xgboost as xgb
from tl import *
D=source_table(); _,qc,_,_=model.features(df); T=target_shared(df,F0,qc)
GRP=sys.argv[1] if len(sys.argv)>1 else 'near'
HI=np.exp(D.ly)>=10; EV=(D.P2>=10)|(D.Qpct>=0.8)
NB=D.grp.isin(['near','turb'])
SRC={'nearbigev':D[NB&EV],'nearbighi':D[NB&HI],'nearbigevhi':D[NB&(EV|HI)],'big':D,'nearbig':D[D.grp.isin(['near','turb'])],'near':D[D.grp=='near'],'all':D,'allhi':D[HI],'allev':D[EV],'nearev':D[(D.grp=='near')&EV],'nearhi':D[(D.grp=='near')&HI]}[GRP]
print(GRP,'source n',len(SRC),'ge25',(np.exp(SRC.ly)>=25).sum())
UCOLS=list(F0.columns)+['lQrel','Qpct']
FX=F0.join(T[['lQrel','Qpct']])          # 내린천2 입력 = 운영 37변수 + 상대유량 2개
MONO2=model.MONO+('lQrel','Qpct','g')
_cache={}
def src_model(cut):
    k=('g',cut)
    if k not in _cache:
        s=SRC[SRC.index<cut]; _cache[k]=model.BagXGB(n=5,mono=('lQrel','Qpct','P0','P2','Pmax3')).fit(s[SHARED],s.yrel.values)
    return _cache[k]
def cut_of(X): return pd.Timestamp(year=X.index.max().year+1,month=1,day=1)

class Stack:
    """A: 원천 모델 예측(g)을 변수 하나로 추가."""
    def fit(self,X,y):
        self.cut=cut_of(X); self.g=src_model(self.cut); self.m=model.Blend(mono=MONO2).fit(self._x(X),y); return self
    def _x(self,X): Z=X.copy(); Z['g']=self.g.predict(T.loc[X.index]); return Z
    def predict(self,X): return self.m.predict(self._x(X))
class Resid:
    """B: 원천 예측을 출발점(사전 정보)으로 두고 내린천2는 차이만 학습."""
    def fit(self,X,y):
        self.g=src_model(cut_of(X)); g=self.g.predict(T.loc[X.index]); self.m=model.Daily().fit(X,y-g); return self
    def predict(self,X): return self.m.predict(X)+self.g.predict(T.loc[X.index])
def pre_booster(cut,seed):
    k=('b',cut,seed)
    if k not in _cache:
        s=SRC[SRC.index<cut]; Z=pd.DataFrame(np.nan,index=range(len(s)),columns=UCOLS)
        for c in SHARED: Z[c]=s[c].values
        mono=tuple(int(c in MONO2) for c in UCOLS)
        p=dict(max_depth=3,eta=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,seed=seed,monotone_constraints=str(mono),nthread=4)
        _cache[k]=(xgb.train(p,xgb.DMatrix(Z,label=s.yrel.values),300),p)
    return _cache[k]
class FineTune:
    """C: 원천에서 XGB를 먼저 키운 뒤(사전학습) 내린천2 자료로 나무를 이어 붙임(미세조정). 운영처럼 Ridge(내린천2만)와 반반."""
    def __init__(self,nft=300,eta=0.05,n=5): self.nft=nft; self.eta=eta; self.n=n
    def fit(self,X,y):
        cut=cut_of(X); self.med=float(np.median(y)); d=xgb.DMatrix(X[UCOLS],label=y-self.med); self.bs=[]
        for sd in range(self.n):
            b0,p=pre_booster(cut,sd); p=dict(p,eta=self.eta)
            self.bs.append(xgb.train(p,d,self.nft,xgb_model=b0.copy()))
        self.r=model.SafeRidge().fit(X[list(F0.columns)],y); return self
    def predict(self,X):
        a=np.mean([b.predict(xgb.DMatrix(X[UCOLS])) for b in self.bs],axis=0)+self.med
        return (a+self.r.predict(X[list(F0.columns)]))/2
class Pool:
    """D: 원천+내린천2 한데 모아 학습(내린천2 가중 w), 내린천2 전용 변수는 원천에서 빈칸."""
    def __init__(self,w=3.0,n=5): self.w=w; self.n=n
    def fit(self,X,y):
        cut=cut_of(X); s=SRC[SRC.index<cut]; self.med=float(np.median(y))
        Z=pd.DataFrame(np.nan,index=range(len(s)),columns=UCOLS)
        for c in SHARED: Z[c]=s[c].values
        XA=pd.concat([Z,X[UCOLS].reset_index(drop=True)],ignore_index=True); yA=np.r_[s.yrel.values,y-self.med]
        wA=np.r_[np.ones(len(s)),np.full(len(y),self.w)]; mono=tuple(int(c in MONO2) for c in UCOLS)
        self.ms=[model.XG(i,monotone_constraints=mono).fit(XA,yA,sample_weight=wA) for i in range(self.n)]
        self.r=model.SafeRidge().fit(X[list(F0.columns)],y); return self
    def predict(self,X): return (np.mean([m.predict(X[UCOLS]) for m in self.ms],axis=0)+self.med+self.r.predict(X[list(F0.columns)]))/2

def flood_lock(lo_oof,lo_te,k=0.75):
    r=np.log(Y[lo_oof.index])-lo_oof; wo=((F0.P2>=10)|(F0.Hh_rng24>=0.3))[lo_oof.index].values; xo=F0.lQpk[lo_oof.index].fillna(2.441).values
    wt=((F0.P2>=10)|(F0.Hh_rng24>=0.3))[lo_te.index].values; xt=F0.lQpk[lo_te.index].fillna(2.441).values; adj=lo_te.copy()
    for st in (False,True):
        c=wo==st; kk=wt==st; base=np.median(r[c])
        sh=HuberRegressor().fit(xo[c].reshape(-1,1),r[c].values).predict(xt[kk].reshape(-1,1)) if st else base
        adj[kk]=lo_te[kk]+base+k*(sh-base)
    return adj
def lock(mk,lo_oof,XX):
    tr=TR&(df.index.year<2024); te=TR&(df.index.year>=2024)
    m,sm=model.fit_log(mk,XX[tr],Y[tr].values); lt=pd.Series(m.predict(XX[te])+sm,index=df.index[te])
    return flood_lock(lo_oof[lo_oof.index.year<2024],lt)

C={'운영':lambda: model.Daily, 'A 변수추가':lambda: Stack,'B 차이학습':lambda: Resid,
   'C 미세조정300':lambda: FineTune,'C 미세조정100':lambda: (lambda: FineTune(nft=100)),
   'D 합쳐학습w1':lambda: (lambda: Pool(w=1)),'D 합쳐학습w3':lambda: Pool,'D 합쳐학습w10':lambda: (lambda: Pool(w=10)),'D 합쳐학습w30':lambda: (lambda: Pool(w=30)),'D 합쳐학습w10x10':lambda: (lambda: Pool(w=10,n=10))}
only=sys.argv[2].split(',') if len(sys.argv)>2 else list(C)
LO={}; LK={}
for n in only:
    t=time.time(); mk=C[n]()
    XX=F0 if n=='운영' else FX
    lo=oof(XX,mk,name=f'{GRP}_{n}'.replace(' ',''))
    LO[n]=flood(lo); fp=f'lock_{GRP}_{n}.pkl'.replace(' ','')
    LK[n]=pd.read_pickle(fp) if os.path.exists(fp) else lock(mk,lo,XX); LK[n].to_pickle(fp)
    print(n,'done',round(time.time()-t),'s',flush=True)
R=report(LO,base='운영') if '운영' in LO else None
print('\n잠금평가 2024–26 (≤2023 학습)')
for n,lp in LK.items():
    q=np.exp(lp); m=metrics(Y[q.index].values,q.values); print(f"{n:>14} n={m['n']} logR2={m['logR2']:.3f} R2={m['R2']:.3f} MAE={m['MAE']:.2f} 비≥25={m['ratio_ge25']:.2f}")
    if n!='운영' and '운영' in LK:
        i=lp.index; print('   vs 운영 ΔlogR2',np.round(block_boot(Y[i].values,np.exp(LK['운영'][i].values),q.values,i,logr2),3),'ΔMAE',np.round(block_boot(Y[i].values,np.exp(LK['운영'][i].values),q.values,i,mae),3))
pd.DataFrame(LO).to_csv(f'../results/t2_{GRP}_oof.csv'); pd.DataFrame(LK).to_csv(f'../results/t2_{GRP}_lock.csv')
