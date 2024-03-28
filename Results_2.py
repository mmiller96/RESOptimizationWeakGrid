import matplotlib.pyplot as plt
import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_directory, 'results')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, "Res2.pdf")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    # Optionally specify the LaTeX preamble to include specific packages
    "text.latex.preamble": r"\usepackage{amsmath}",
})
size = 30

categories = [' ', ' ', ' ']
y1=np.array([3.57,2.72,0.54])
y2=np.array([1.32,0.53,0.08])
y3=np.array([0.15,0.00,0.00])

def format_y_tick(value, tick_number):
    return f'${value:.2f}'


plt.figure(figsize=(10,6))
bar_width = 0.2

# To calculate the index for every category
indices = np.arange(len(categories))

# To create bars for the first series
plt.bar(indices - bar_width, y1, bar_width, edgecolor='black', alpha=0.7, color='grey', label='No placement')
plt.bar(indices, y2, bar_width, edgecolor='black', alpha=0.7, color='blue', label='Only PVs')
plt.bar(indices + bar_width, y3, bar_width, edgecolor='black', alpha=0.7, color='green', label='PVs with hydrogen')


# To add the categories labels in x axis
plt.xticks(indices, categories)

# To add leyends 
plt.legend(fontsize=size*0.55)
plt.grid(True)
#plt.gca().get_yaxis().set_major_formatter(FuncFormatter(format_y_tick))
plt.ylabel('Load curtailment cost [k$\$$/h]', fontsize=size*0.8)
plt.xticks(range(len(y1)), ["1.25 MW DG Outage", "1 MW DG Outage", "0.6 MW DG Outage"], fontsize=size*0.7)
plt.yticks(fontsize=size*0.8)
# To show the combined bars diagram
plt.tight_layout()
plt.savefig(file_path)
plt.show()