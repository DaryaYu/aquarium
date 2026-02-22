import numpy as np
from tqdm import tqdm


class BPR_Opt:
    def __init__(self, n_users, n_items, factors=20, lr=0.01, reg=0.01):
        self.W = np.random.normal(0, 0.01, (n_users, factors))
        self.H = np.random.normal(0, 0.01, (n_items, factors))
        self.lr = lr
        self.reg = reg
        self.n_items = n_items

    def fit(self, train_dict, n_epochs=10):
        users = list(train_dict.keys())
        n_samples = sum(len(v) for v in train_dict.values())
        
        for epoch in range(n_epochs):
            # Bootstrap sampling of triplets
            for _ in tqdm(range(n_samples), desc=f"Epoch {epoch+1}"):
                # u: user, i: positive item (rating >= 4), j: negative / unobserved item
                u = np.random.choice(users)
                i = np.random.choice(list(train_dict[u]))
                j = np.random.randint(0, self.n_items)
                while j in train_dict[u]:
                    j = np.random.randint(0, self.n_items)

                # Predict score difference
                x_uij = np.dot(self.W[u], self.H[i]) - np.dot(self.W[u], self.H[j])
                
                # Gradient of BPR-Opt: 1 / (1 + exp(x_uij))
                grad = 1.0 / (1.0 + np.exp(x_uij))
                
                # Update Latent Vectors
                self.W[u] += self.lr * (grad * (self.H[i] - self.H[j]) - self.reg * self.W[u])
                self.H[i] += self.lr * (grad * self.W[u] - self.reg * self.H[i])
                self.H[j] += self.lr * (grad * (-self.W[u]) - self.reg * self.H[j])

    def recommend(self, user_id, train_dict, k=10, user_map=None, item_map=None, test=None):
        if user_map:
            user_id = user_map[user_id]
        
        scores = np.dot(self.H, self.W[user_id])

        # Exclude training items:
        if user_id in train_dict:
            scores[list(train_dict[user_id])] = -np.inf 
        
        recommended = np.argsort(scores)[::-1][:k]
        
        if item_map:
            item_map_reverse = {v: k for k, v in item_map.items()}
            vectorizer = np.vectorize(item_map_reverse.get)
            recommended = vectorizer(recommended)
        
        return recommended

