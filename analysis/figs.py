import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rep=json.load(open("out/report.json"))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.edgecolor":"#888","axes.linewidth":0.6,
                     "xtick.color":"#52514e","ytick.color":"#0b0b0b","axes.labelcolor":"#0b0b0b"})
INK2="#52514e"; GRID="#e6e5e0"; BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"
# ---- Fig 1: constrained accuracy with 95% CI
A=sorted(rep["accuracy_constrained"], key=lambda a:a["acc"])
g=rep["gemini_acc"]
fig,ax=plt.subplots(figsize=(3.5,2.55),dpi=300)
ys=range(len(A))
for y,a in zip(ys,A):
    ax.plot([a["lo"]*100,a["hi"]*100],[y,y],color=BLUE,lw=1.2,solid_capstyle="round",zorder=2)
    ax.scatter([a["acc"]*100],[y],s=16,color=BLUE,edgecolor="white",linewidth=0.8,zorder=3)
    ax.text(a["hi"]*100+1.2,y,f'{a["acc"]*100:.1f}',va="center",fontsize=6.5,color=INK2)
yg=len(A)+0.4
ax.plot([g["lo"]*100,g["hi"]*100],[yg,yg],color=ORANGE,lw=1.2,zorder=2)
ax.scatter([g["acc"]*100],[yg],s=16,marker="D",color=ORANGE,edgecolor="white",linewidth=0.8,zorder=3)
ax.text(g["lo"]*100-1.2,yg,f'{g["acc"]*100:.1f}',va="center",ha="right",fontsize=6.5,color=INK2)
lab=[a["model"]+("†" if a["model"]=="qwen3:8b" else "") for a in A]+["Gemini Flash (cloud)"]
ax.set_yticks(list(ys)+[yg]); ax.set_yticklabels(lab)
ax.axvline(10,color=INK2,lw=0.7,ls=(0,(3,2)),zorder=1)
ax.text(10.8,-0.9,"chance ≈10%",fontsize=6,color=INK2,va="center")
ax.set_xlim(0,100); ax.set_ylim(-1.4,yg+0.8)
ax.set_xlabel("Accuracy, % (95% CI), 200 MMLU-Pro CS items")
ax.grid(axis="x",color=GRID,lw=0.5); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig("out/fig1_accuracy.png",dpi=300); plt.close(fig)
# ---- Fig 2: seconds per item by machine
order=[a["model"] for a in A if a["model"]!="qwen3:8b"]
S={(s["tier"],s["model"]):s for s in rep["speed"]}
tiers=[("homelab","On-prem i5-6500 (2015), 4 GB",BLUE,"o",-0.27),("vps","Cloud VPS, 4 vCPU, ~4.4 GB",ORANGE,"s",-0.09),("laptop","Laptop i5-1235U, WSL",AQUA,"^",0.09),("pi","Raspberry Pi 4, 4 GB (ARM)","#52514e","D",0.27)]
fig,ax=plt.subplots(figsize=(3.5,2.9),dpi=300)
for t,name,c,mk,off in tiers:
    pts=[(S[(t,m)]["a_wall_med"],i+off) for i,m in enumerate(order) if (t,m) in S]
    xs=[p[0] for p in pts]
    ax.scatter(xs,[p[1] for p in pts],s=14,marker=mk,color=c,edgecolor="white",linewidth=0.6,label=name,zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("Median seconds per exam item (log scale)")
ax.set_xscale("log"); ax.set_xlim(1,100)
ax.set_xticks([1,2,5,10,20,50,100]); ax.set_xticklabels(["1","2","5","10","20","50","100"]); ax.grid(axis="x",color=GRID,lw=0.5); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.legend(loc="lower center",bbox_to_anchor=(0.42,1.0),ncol=1,fontsize=6,frameon=False,handletextpad=0.3)
fig.tight_layout(); fig.savefig("out/fig2_speed.png",dpi=300); plt.close(fig)
print("ok")
