import matplotlib.pyplot as plt
import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_directory, 'results')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, "Res1.pdf")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 26

y1=np.array([0.00,12.54,13.06])
y2=np.array([0.00,0.00,1.19])
y3=np.array([0.00,0.00,2.60])
y4=np.array([11.01,3.35,0.12])
y5=np.array([112.06,57.05,56.76])
y6=np.array([33.66,17.28,15.05])


plt.figure(figsize=(10,6))
plt.bar(range(len(y1)), y1, edgecolor='black', alpha=0.7, color="blue",label="PV cost (Investment, O$\&$M)",width=0.2)
plt.bar(range(len(y1)), y2, edgecolor='black', alpha=0.7, bottom=y1,color="green",label="Electrolyzer cost (Investment, O$\&$M)", width=0.2)
plt.bar(range(len(y1)), y3, edgecolor='black', alpha=0.7, bottom=y1+y2,color="yellow",label="Fuel cells cost (Investment, O$\&$M)", width=0.2)
plt.bar(range(len(y1)), y4, edgecolor='black', alpha=0.7, bottom=y1 + y2 + y3, color="red", label="Load curtaliment cost", width=0.2)
plt.bar(range(len(y1)), y5, edgecolor='black', alpha=0.7,bottom=y1 + y2 + y3 + y4, color="grey", label="External grid cost", width=0.2)
plt.bar(range(len(y1)), y6, edgecolor='black', alpha=0.7, bottom=y1 + y2 + y3 + y4 + y5, color="black", label="Diesel fuel cost", width=0.2)


plt.legend(fontsize=size*0.55)
#plt.xlabel('Configurations', fontsize=size)
plt.ylabel('Present value cost [M$\$$]', fontsize=size)
plt.xticks(range(len(y1)), ["Without new technologies", "Only with PV allocation", "PV and H2 allocation"], fontsize=size*0.8)
plt.yticks(fontsize=size*0.8)
# 1qplt.xticks(["without new technologies","Only with PV allocation","WithPVs and Hydrogen devices allocation"])
plt.grid(True)
plt.tight_layout()
plt.savefig(file_path)
plt.show()
