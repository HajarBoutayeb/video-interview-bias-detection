import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

# Define stages of the pipeline
stages = [
    ("Prétraitement\n(Chargement, segmentation)", 0.5),
    ("Extraction\nmultimodale", 2.5),
    ("Compilation\n(Agrégation)", 4.5),
    ("Validation\n(Cohérence)", 6.5),
    ("Sauvegarde\n(CSV structuré)", 8.5)
]

# Draw boxes and arrows
for text, x in stages:
    box = FancyBboxPatch((x, 1), 1.8, 1, boxstyle="round,pad=0.2",
                         edgecolor="black", facecolor="#cfe2f3")
    ax.add_patch(box)
    ax.text(x+0.9, 1.5, text, ha="center", va="center", fontsize=9)
    
# Draw arrows between boxes
for i in range(len(stages)-1):
    ax.annotate("", xy=(stages[i+1][1], 1.5), xytext=(stages[i][1]+1.8, 1.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))

plt.title("Figure 3.7 – Pipeline d'extraction intégré", fontsize=12, pad=20)
plt.show()
