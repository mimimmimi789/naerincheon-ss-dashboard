"""새 후보 평가 틀: 2016–26 연도별 전진검증 OOF → 홍수일 보정 ¾(그해 이전 OOF로만 적합) → 2019–26 및 2026(=잠금평가) 지표."""
from h import *
from sklearn.linear_model import HuberRegressor
df=pd.read_pickle('df.pkl'); F0=pd.read_pickle('F.pkl'); TR=obs_mask(df); Y=df.SS
NR4=pd.to_datetime(['2019-06-18','2021-03-03','2023-06-13','2024-06-25'])
NONRUN=((df.SS>=10)&(F0.dHmax<0.5))|df.index.isin(NR4)

def oof(F, mk=model.Daily, fit=None, years=range(2016,2027), name=None):
    """fit(mk,X,y,df_rows) -> (model, shift) ; 기본 model.fit_log. 결과는 log 예측(보정 전)."""
    if name and os.path.exists(f'oof_{name}.pkl'): return pd.read_pickle(f'oof_{name}.pkl')
    fit = fit or (lambda mk,X,y,idx: model.fit_log(mk,X,y))
    out=pd.Series(np.nan,index=df.index)
    for yr in years:
        tr=TR&(df.index.year<yr); te=TR&(df.index.year==yr)
        m,sm=fit(mk,F[tr],Y[tr].values,df.index[tr]); out[te]=m.predict(F[te])+sm
    out=out[TR&(df.index.year>=min(years))]
    if name: out.to_pickle(f'oof_{name}.pkl')
    return out

def flood(lo, F=F0, k=0.75, start=2019):
    """lo: log OOF. 그해 이전 OOF 잔차로 홍수일 Huber 선형(¾)·평상시 중앙값."""
    r=np.log(Y[lo.index])-lo; wet=((F.P2>=10)|(F.Hh_rng24>=0.3))[lo.index].values; x=F.lQpk[lo.index].fillna(2.441).values
    adj=lo.copy()
    for yr in range(start,int(lo.index.year.max())+1):
        te=lo.index.year==yr; cal=lo.index.year<yr
        for st in (False,True):
            c=cal&(wet==st); kk=te&(wet==st)
            if not kk.any(): continue
            base=np.median(r[c])
            sh=HuberRegressor().fit(x[c].reshape(-1,1),r[c].values).predict(x[kk].reshape(-1,1)) if st else base
            adj[kk]=lo[kk]+base+k*(sh-base)
    return adj[adj.index.year>=start]

def score(lp, name=''):
    rows={}
    for lab,(a,b) in {'19-26':(2019,2026),'2026':(2026,2026)}.items():
        q=np.exp(lp[(lp.index.year>=a)&(lp.index.year<=b)])
        if not len(q): continue
        yy=Y[q.index]; m=metrics(yy.values,q.values); run=~NONRUN[q.index]
        rows[lab]=dict(n=m['n'],logR2=m['logR2'],R2=m['R2'],MAE=m['MAE'],MAE_lt10=m['MAE_lt10'],
            r10run=np.median((q/yy)[(yy>=10)&run]) if ((yy>=10)&run).any() else np.nan, ratio25=m['ratio_ge25'],MAE25=m['MAE_ge25'],
            a25=f"{int(((q>=25)&(yy>=25)).sum())}/{int((q>=25).sum())}/{int((yy>=25).sum())}")
    return rows

def report(C, base='base'):
    out=[]
    for n,lp in C.items():
        s=score(lp)
        for lab,d in s.items(): out.append(dict(cand=n,period=lab,**d))
    R=pd.DataFrame(out); pd.set_option('display.width',250); print(R.round(3).to_string(index=False))
    b=C[base]; idx=b.index[b.index.year>=2019]
    for n,lp in C.items():
        if n==base: continue
        i=idx.intersection(lp.index); yv=Y[i].values
        print(f'{n:>14} vs {base}: ΔlogR2',np.round(block_boot(yv,np.exp(b[i].values),np.exp(lp[i].values),i,logr2),3),
              'ΔMAE',np.round(block_boot(yv,np.exp(b[i].values),np.exp(lp[i].values),i,mae),3))
    return R
