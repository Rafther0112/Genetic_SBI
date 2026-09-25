import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family":"serif","font.size":8})
col={"exact":"#0f6e56","nb":"#2b6cb0","lna":"#7e4ea8","cle":"#ba7517","hyb":"#3a9e8f"}
# Table 2 (telegraph, k_on, mean±std over five seeds)
tel_lab=["[0,1.1]","[1.1,1.5]","[1.5,2.3]","[2.3,4.6]","[4.6,12.5]","[12.5,146]"]
T={"CME (exact)":("exact","o",[.92,.91,.90,.90,.88,.92],[.02,.02,.01,.04,.04,.04]),
   "NB":("nb","D",[.92,.92,.92,.91,.90,.89],[.02,.02,.02,.03,.02,.05]),
   "hybrid":("hyb","v",[.79,.81,.85,.83,.83,.92],[.01,.01,.03,.02,.03,.01]),
   "LNA":("lna","^",[.73,.66,.63,.50,.35,.21],[.03,.03,.02,.06,.02,.04]),
   "CLE":("cle","s",[.58,.31,.13,.06,.01,.00],[.04,.06,.03,.01,.00,.00])}
# Table 3 (three-state, k_syn, pooled over three seeds)
ts_lab=["[0,1.1]","[1.1,1.9]","[1.9,3.3]","[3.3,7.1]","[7.1,19.6]","[19.6,175]"]
S={"exact (FSP)":("exact","o",[.95,.92,.88,.91,.93,.94]),
   "hybrid":("hyb","v",[.96,.93,.88,.91,.93,.94]),
   "NB":("nb","D",[.93,.91,.80,.73,.68,.69]),
   "CLE":("cle","s",[.05,.27,.33,.33,.28,.11])}
fig,ax=plt.subplots(1,2,figsize=(6.8,2.45),sharey=True)
x=np.arange(6)
for name,(c,m,v,e) in T.items():
    ls="--" if c=="hyb" else "-"
    ax[0].errorbar(x,v,yerr=e,fmt=m+ls,color=col[c],ms=3.5,lw=1.2,capsize=2,label=name)
for name,(c,m,v) in S.items():
    ls="--" if c=="hyb" else "-"
    ax[1].plot(x,v,m+ls,color=col[c],ms=3.5,lw=1.2,label=name)
for a,lab,t in [(ax[0],tel_lab,r"A  telegraph, $k_{\mathrm{on}}$ (five seeds)"),(ax[1],ts_lab,r"B  three-state promoter, $k_{\mathrm{syn}}$ (three seeds, pooled)")]:
    a.axhline(0.9,ls=":",color="gray",lw=1)
    a.set_xticks(x); a.set_xticklabels(lab,fontsize=6.5)
    a.set_xlabel("empirical Fano factor (sextile range)")
    a.set_title(t,fontsize=8,loc="left"); a.set_ylim(-0.03,1.02)
    a.spines[["top","right"]].set_visible(False)
ax[0].set_ylabel("90% interval coverage")
h,l=ax[0].get_legend_handles_labels()
l=[ "exact (CME / FSP)" if "CME" in t else t for t in l]
fig.legend(h,l,fontsize=7,frameon=False,loc="upper center",ncol=5,bbox_to_anchor=(0.5,1.0))
plt.tight_layout(pad=0.3,rect=(0,0,1,0.9)); plt.savefig("coverage_sextiles.pdf")
