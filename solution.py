import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "target"]
raw_data = pd.read_csv(data_url, header=None, names=cols)

X = raw_data.iloc[:, :4].to_numpy()
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
y = raw_data["target"].map(label_map).to_numpy()

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=200, seed=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None

    def fit_predict(self, data):
        np.random.seed(self.seed)
        start_idx = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[start_idx]

        for _ in range(self.max_iter):
            dists = np.linalg.norm(data[:, None] - self.centroids, axis=2)
            preds = np.argmin(dists, axis=1)

            new_centers = []
            for idx in range(self.k):
                pts = data[preds == idx]
                if len(pts) > 0:
                    new_centers.append(pts.mean(axis=0))
                else:
                    new_centers.append(self.centroids[idx])
            new_centers = np.array(new_centers)

            if np.allclose(self.centroids, new_centers, atol=1e-4):
                break
            self.centroids = new_centers

        return preds

model = KMeansClustering(n_clusters=3)
raw_clusters = model.fit_predict(X)

final_preds = np.zeros_like(raw_clusters)
for c in range(3):
    sub_labels = y[raw_clusters == c]
    if len(sub_labels) > 0:
        counts = np.bincount(sub_labels)
        best_fit = np.argmax(counts)
        final_preds[raw_clusters == c] = best_fit

correct = np.sum(final_preds == y)
acc = (correct / len(y)) * 100

print(f"Total Samples Tested: {len(y)}")
print(f"Correct Matches: {correct}")
print(f"Calculated Accuracy: {acc:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(X[:, 2], X[:, 3], c=y, cmap="viridis", edgecolors="k", s=35)
ax1.set_title("Ground Truth Labels")
ax1.set_xlabel("Petal Length")
ax1.set_ylabel("Petal Width")

ax2.scatter(X[:, 2], X[:, 3], c=final_preds, cmap="viridis", edgecolors="k", s=35)
ax2.scatter(model.centroids[:, 2], model.centroids[:, 3], c="red", marker="X", s=120)
ax2.set_title(f"K-Means Predicted (Acc: {acc:.2f}%)")
ax2.set_xlabel("Petal Length")
ax2.set_ylabel("Petal Width")

plt.tight_layout()
plt.savefig("output.png")
plt.show()