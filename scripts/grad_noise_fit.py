"""B_simple 离线打分 — 对齐 knowledge/training_optimization_analyze/grad_noise_scale_B_simple.md
口径: MoM8 统计量; 可辨识 → 逐对(B≥4)比值中位 + bootstrap 90% CI (+WLS 副读数);
不可辨识(退火态) → 只报 ‖G‖²上界=min_B MoM8(S) 与 B_simple 下界=tr/上界。
诊断: ①高档逐对截距负号 ②折半比 S(B)/S(2B)>2 越界 ③三估计量摆动 ④B=1 重尾 max/med。"""
import json, sys, itertools, numpy as np
OUT='/home/z50057756/code/RnG_lagernvs/experiments/grad_noise_probe'
def load(p):
    d=json.load(open(f'{OUT}/{p}.json'))['bsizes']
    try: d={**d,**json.load(open(f'{OUT}/{p}_ext.json'))['bsizes']}
    except FileNotFoundError: pass
    return {int(k):np.array(v['sqnorms']) for k,v in d.items()}
def mom(a,g=8):
    n=len(a)//g; return np.median([a[i*n:(i+1)*n].mean() for i in range(g)]) if n>0 else a.mean()
def trim(a,f=0.10):
    s=np.sort(a); return s[:int(len(s)*(1-f))].mean()
def pair_fit(S,Bs):
    G2s,trs,rats=[],[],[]
    for Bi,Bj in itertools.combinations(Bs,2):
        G2=(Bj*S[Bj]-Bi*S[Bi])/(Bj-Bi); tr=(S[Bi]-S[Bj])*Bi*Bj/(Bj-Bi)
        G2s.append(G2);trs.append(tr);rats.append(tr/G2 if G2>0 else np.nan)
    return np.array(G2s),np.array(trs),np.array(rats)
def score(p):
    d=load(p); Bs=sorted(d); S={B:mom(d[B]) for B in Bs}
    hi=[b for b in Bs if b>=16]
    G2hi,_,_=pair_fit(S,hi)
    neg_hi=float((G2hi<0).mean())
    folds=[(B,S[B]/S[2*B]) for B in Bs if 2*B in S]
    fold_viol=[(B,r) for B,r in folds if r>2.0]
    ests={}
    for name,st in (('mom8',mom),('trim10',trim),('median',np.median)):
        Sx={B:st(d[B]) for B in Bs}
        _,trx,ratx=pair_fit(Sx,[b for b in Bs if b>=4])
        ests[name]=np.nanmedian(ratx)
    sway=max(ests.values())/max(1e-9,min(ests.values())) if min(ests.values())>0 else np.inf
    tail=float(d[1].max()/np.median(d[1])) if 1 in d else np.nan
    _,trs4,rats4=pair_fit(S,[b for b in Bs if b>=4])
    tr_med=np.median(trs4)
    # 审计判据: bootstrap 下逐对(B≥4)中位 G2 为非正的概率
    rng=np.random.default_rng(1); negG=0; boots=[]; bootG=[]
    for _ in range(400):
        Sx={B:mom(d[B][rng.integers(0,len(d[B]),len(d[B]))]) for B in Bs}
        Gx,_,rx=pair_fit(Sx,[b for b in Bs if b>=4])
        g=np.median(Gx); bootG.append(g)
        if g<=0: negG+=1
        boots.append(np.nanmedian(rx))
    pneg=negG/400
    identifiable = pneg<0.05 and neg_hi==0 and len(fold_viol)<=2 and sway<2.0 and np.isfinite(np.nanmedian(rats4)) and np.nanmedian(rats4)>0
    row={'point':p,'S':S,'tr':tr_med,'tail':tail,'neg_hi':neg_hi,'p_negG2':pneg,
         'fold_viol':fold_viol,'sway':round(float(sway),2),'ests':{k:round(float(v),1) for k,v in ests.items()}}
    ub=min(S[B] for B in Bs if B>=32)
    row.update(G2_ub=float(ub), B_simple_lb=float(tr_med/ub))
    if identifiable:
        bs=np.nanmedian(rats4)
        row.update(mode='POINT', B_simple=float(bs), ci=[float(np.percentile(boots,5)),float(np.percentile(boots,95))])
    else:
        row.update(mode='BOUND')
    return row


def score_xip(p):
    """xip 模式: <point>_xip.json。G2=mean(xips) 精确无偏; tr=B/4*mean(d2s)。
    P(G2<=0)<0.05 → 点估计+bootstrap 90% CI; 否则单侧界: G2_ub=mean+1.645*SEM,
    tr_lb=mean-1.645*SEM → B_simple > tr_lb/G2_ub。"""
    d=json.load(open(f'{OUT}/{p}_xip.json'))
    rows=[]
    for B,v in d['bsizes'].items():
        B=int(B); x=np.array(v['xips']); d2=np.array(v['d2s']); n=len(x)
        G2=x.mean(); sem=x.std(ddof=1)/np.sqrt(n)
        tr=B/4*d2.mean(); tr_sem=B/4*d2.std(ddof=1)/np.sqrt(len(d2))
        rng=np.random.default_rng(1)
        pneg=float(np.mean([x[rng.integers(0,n,n)].mean()<=0 for _ in range(1000)]))
        row={'point':p,'B':B,'n':n,'G2':float(G2),'G2_sem':float(sem),'tr':float(tr),'p_negG2':pneg}
        if pneg<0.05:
            bs=[]
            for _ in range(2000):
                g=x[rng.integers(0,n,n)].mean(); t=B/4*d2[rng.integers(0,len(d2),len(d2))].mean()
                if g>0: bs.append(t/g)
            row.update(mode='POINT', B_simple=float(tr/G2),
                       ci=[float(np.percentile(bs,5)),float(np.percentile(bs,95))])
        else:
            ub=G2+1.645*sem; lb_tr=max(0.0,tr-1.645*tr_sem)
            row.update(mode='BOUND', G2_ub=float(ub), B_simple_lb=float(lb_tr/ub) if ub>0 else float('inf'))
        rows.append(row)
    return rows

if __name__=='__main__':
    args=sys.argv[1:]
    if args and args[0]=='--xip':
        for p in args[1:]:
            for r in score_xip(p):
                if r['mode']=='POINT':
                    print(f"{r['point']:>20} B={r['B']:>3} (n={r['n']}): B_simple={r['B_simple']:6.1f} [{r['ci'][0]:.1f},{r['ci'][1]:.1f}]  G2={r['G2']:.4f}±{r['G2_sem']:.4f} tr={r['tr']:.3f}")
                else:
                    print(f"{r['point']:>20} B={r['B']:>3} (n={r['n']}): G2 与 0 不可分 (P={r['p_negG2']:.2f}) → G2<{r['G2_ub']:.4f}, B_simple>{r['B_simple_lb']:.0f}  tr={r['tr']:.3f}")
        sys.exit(0)
    for p in args:
        r=score(p)
        diag=f"负截距(高档对){r['neg_hi']*100:.0f}% 折半越界{len(r['fold_viol'])} 摆动×{r['sway']} 重尾{r['tail']:.0f}"
        diag=f"P(G2≤0)={r['p_negG2']:.3f} "+diag
        if r['mode']=='POINT':
            print(f"{r['point']:>20}: B_simple={r['B_simple']:6.1f} [{r['ci'][0]:.1f},{r['ci'][1]:.1f}] (下界{r['B_simple_lb']:.0f})  tr={r['tr']:.3f} | {diag}")
        else:
            print(f"{r['point']:>20}: 不可辨识→ ‖G‖²<{r['G2_ub']:.4f}  B_simple>{r['B_simple_lb']:.0f}  tr={r['tr']:.3f} | {diag}")
