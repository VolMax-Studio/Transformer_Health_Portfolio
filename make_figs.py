import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open("verified_results.json"))
fig,ax=plt.subplots(1,3,figsize=(15,4.5))

# Panel 1: uplift decomposition
p=R["phase2"]
vals=[p["duval_3gas"],p["rf_3gas"],p["rf_5gas"]]
labs=["Duval 1\n(3 gas, rules)","RF\n(3 gas)","RF\n(5 gas+ratios)"]
cols=["#b0413e","#e08e45","#3d7a5a"]
b=ax[0].bar(labs,vals,color=cols,width=0.6)
ax[0].set_ylim(0,1); ax[0].set_ylabel("fault-type accuracy (5-fold CV)")
ax[0].set_title("Step 2: uplift decomposition",fontweight='bold')
for r,v in zip(b,vals): ax[0].text(r.get_x()+r.get_width()/2,v+0.02,f"{v:.3f}",ha='center',fontweight='bold')
ax[0].annotate(f"+{p['algo_gain']:.3f}\nalgorithm",xy=(0.5,0.58),ha='center',fontsize=9,color="#7a3a00")
ax[0].annotate(f"+{p['info_gain']:.3f}\nextra gases",xy=(1.5,0.75),ha='center',fontsize=9,color="#1d4a30")

# Panel 2: reliability diagram
mc=[b[0] for b in R["calibration"]["bins"]]; ea=[b[1] for b in R["calibration"]["bins"]]
ax[1].plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
ax[1].plot(mc,ea,'o-',color="#3d7a5a",markersize=8,label='RF (out-of-fold)')
ax[1].fill_between([0,1],[0,1],[0,0],alpha=0.04,color='red')
ax[1].set_xlim(0.3,1); ax[1].set_ylim(0.3,1); ax[1].set_xlabel("mean predicted confidence")
ax[1].set_ylabel("empirical accuracy"); ax[1].legend(loc='upper left',fontsize=8)
ax[1].set_title(f"Step 3: calibration (ECE={R['calibration']['ece']}, under-confident)",fontweight='bold')
ax[1].text(0.62,0.42,"points ABOVE diagonal\n= under-confident\n= safe for production",fontsize=8,color="#1d4a30")

# Panel 3: boundary vs interior
bd=R["boundary"]
x=np.arange(2); w=0.35
ax[2].bar(x-w/2,[bd["rf_boundary"],bd["rf_interior"]],w,label='Random Forest',color="#3d7a5a")
ax[2].bar(x+w/2,[bd["duval_boundary"],bd["duval_interior"]],w,label='Duval 1',color="#b0413e")
ax[2].set_xticks(x); ax[2].set_xticklabels([f"boundary\n(n={bd['n_boundary']})",f"interior\n(n={bd['n_interior']})"])
ax[2].set_ylabel("accuracy"); ax[2].set_ylim(0,1); ax[2].legend(fontsize=8)
ax[2].set_title("Step 3: where Duval breaks (decision geometry)",fontweight='bold')
for i,(r,d) in enumerate([(bd["rf_boundary"],bd["duval_boundary"]),(bd["rf_interior"],bd["duval_interior"])]):
    ax[2].text(i-w/2,r+0.02,f"{r:.2f}",ha='center',fontsize=9); ax[2].text(i+w/2,d+0.02,f"{d:.2f}",ha='center',fontsize=9)

plt.tight_layout(); plt.savefig("fig_results.png",dpi=110,bbox_inches='tight')
print("saved fig_results.png")
