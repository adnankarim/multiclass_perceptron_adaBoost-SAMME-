import numpy as np

class Perceptron:
    """
    Multi-class Perceptron that respects sample weights in its update.
    For each misclassified sample i, the update is scaled by sample_weight[i].
    """

    def __init__(self, n_classes=10, n_features=64, eta=1.0, max_iter=10):
        """
        Parameters
        ----------
        n_classes : int
            Number of classes (K).
        n_features : int
            Dimension of the feature space (d).
        eta : float, optional
            Learning rate (default 1.0).
        max_iter : int, optional
            Number of epochs (passes over the data) per fit() call (default=10).
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.eta = eta
        self.max_iter = max_iter

        # Initialize weight matrix and bias for each class
        self.W = np.zeros((self.n_classes, self.n_features))
        self.b = np.zeros(self.n_classes)

    def fit(self, X, y, sample_weight=None):
        """
        Train the Weighted Perceptron.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Class labels in {0, 1, ..., n_classes-1}.
        sample_weight : ndarray of shape (n_samples,) or None
            Sample weights. If None, use uniform weights (1.0).
        """
        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)

        for _ in range(self.max_iter):
            errors = 0
            # One full pass (epoch) over data
            for i, x_i in enumerate(X):
                scores = self.W.dot(x_i) + self.b  # shape (n_classes,)
                pred = np.argmax(scores)
                true_label = y[i]

                if pred != true_label:
                    # Weighted update
                    w_i = self.eta * sample_weight[i]
                    # Increase weights for the true label
                    self.W[true_label] += w_i * x_i
                    self.b[true_label] += w_i
                    # Decrease weights for the predicted label
                    self.W[pred] -= w_i * x_i
                    self.b[pred] -= w_i
                    errors += 1

            # Early stop if no errors this epoch
            if errors == 0:
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels in {0, 1, ..., n_classes-1}.
        """
        y_pred = []
        for x_i in X:
            scores = self.W.dot(x_i) + self.b
            pred = np.argmax(scores)
            y_pred.append(pred)
        return np.array(y_pred)
