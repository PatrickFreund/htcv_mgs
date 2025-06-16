import matplotlib.pyplot as plt

f1_scores = {
    'A': 0.8570,
    'B': 0.7845,
    'IN': 0.7090,
    'K': 0.7886,
    'KXN': 0.8394
}

subdatasets = list(f1_scores.keys())
scores = list(f1_scores.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(subdatasets, scores, color='orange')
plt.ylim(0, 1)

plt.title("F1 Score by Experiments")
plt.xlabel("Experiments with K-Fold Split")
plt.ylabel("F1 Score")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height - 0.05, f"{height:.3f}", ha='center', color='black')

plt.show()