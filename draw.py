import matplotlib.pyplot as plt

# 数据：Accuracy（与之前表一致）
T = [50,100,200,300,500]
acc = {
    "Adult":[0.82,0.84,0.85,0.86,0.88],
    "Credit":[0.74,0.76,0.78,0.80,0.82],
    "HR":[0.70,0.73,0.75,0.77,0.78],
    "Nursery":[0.80,0.84,0.86,0.87,0.88]
}
M = [1,3,5,7,10]
accM = {
    "Adult":[0.82,0.84,0.86,0.88,0.89],
    "Credit":[0.74,0.78,0.80,0.82,0.84],
    "HR":[0.72,0.75,0.77,0.79,0.80],
    "Nursery":[0.82,0.85,0.87,0.89,0.90]
}

# 1) Accuracy vs T
plt.figure(figsize=(4,2.2))
for k in ["Adult","Credit","HR","Nursery"]:
    plt.plot(T,[100*x for x in acc[k]], marker='o', linewidth=1)
plt.xlabel("Monte Carlo samples T")
plt.ylabel("Accuracy (%)")
plt.legend(["Adult","Credit","HR","Nursery"], ncol=2, fontsize=8, loc='upper center', bbox_to_anchor=(0.5,1.35))
plt.tight_layout()
plt.savefig("fig_sensitivity_T.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close()

# 2) Accuracy vs M
plt.figure(figsize=(4,2.2))
for k in ["Adult","Credit","HR","Nursery"]:
    plt.plot(M,[100*x for x in accM[k]], marker='o', linewidth=1)
plt.xlabel("Cleaning budget M (%)")
plt.ylabel("Accuracy (%)")
plt.legend(["Adult","Credit","HR","Nursery"], ncol=2, fontsize=8, loc='upper center', bbox_to_anchor=(0.5,1.35))
plt.tight_layout()
plt.savefig("fig_sensitivity_M.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close()
