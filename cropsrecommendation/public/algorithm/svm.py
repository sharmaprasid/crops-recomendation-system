import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100, batch_size=32):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros((len(self.classes), n_features))
        self.bias = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            y_binary = np.where(y == c, 1, -1)
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.n_iters):
                indices = np.random.choice(
                    n_samples, self.batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y_binary[indices]
                scores = np.dot(X_batch, w) - b
                margins = y_batch * scores
                misclassified = margins < 1
                grad_w = self.lambda_param * w - \
                    np.dot(X_batch.T, y_batch * misclassified) / \
                    self.batch_size
                grad_b = -np.sum(y_batch * misclassified) / self.batch_size
                w -= self.lr * grad_w
                b -= self.lr * grad_b
            self.weights[i] = w
            self.bias[i] = b

    def predict(self, X):
        output = np.dot(X, self.weights.T) - self.bias
        return self.classes[np.argmax(output, axis=1)]
