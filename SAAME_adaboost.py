import numpy as np
import matplotlib.pyplot as plt

class SAMME_Adaboost:
    """
    Multi-class AdaBoost (SAMME) that calls a Perceptron
    or any other learner that can handle sample_weight.
    
    - No bootstrapping: the full dataset is used each iteration.
    - Sample weights are updated after each round and passed to the learner.
    """

    def __init__(self, n_rounds=50, base_learner_factory=None, n_classes=None):
        """
        Parameters
        ----------
        n_rounds : int
            Number of boosting rounds.
        base_learner_factory : callable
            A function returning an *unfitted* base learner supporting 
            fit(X, y, sample_weight=...).
        n_classes : int, optional
            Number of classes. If None, inferred from data.
        """
        self.n_rounds = n_rounds
        self.base_learner_factory = base_learner_factory
        self.n_classes = n_classes
        
        # For storing alpha^(m) and the trained weak models
        self.alphas = []
        self.models = []

        # For tracking metrics
        self.train_accuracies = []
        self.train_losses = []
        self.validation_accuracies = []
        self.validation_losses = []
        self.iterations = []

    def _aggregate_scores(self, X,N=None):
        """
        Aggregate the weighted scores from all weak learners.
        Shape: (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        for alpha_m, model in zip(self.alphas if not N  else self.alphas[: N+1],self.models if not N else self.models[:N+1]):
            y_pred = model.predict(X)
            for i in range(n_samples):
                scores[i, y_pred[i]] += alpha_m
        return scores

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Fit the SAMME Adaboost model on the entire dataset each iteration.

        Parameters
        ----------
        X_train : ndarray (n_samples, n_features)
        y_train : ndarray (n_samples,)
        X_valid, y_valid : optional validation set
        """
        n_samples = X_train.shape[0]

        if self.n_classes is None:
            self.n_classes = len(np.unique(y_train))

        # Initialize sample weights
        w = np.ones(n_samples) / n_samples

        # Reset stored attributes
        self.alphas.clear()
        self.models.clear()
        self.train_accuracies.clear()
        self.train_losses.clear()
        self.validation_accuracies.clear()
        self.validation_losses.clear()
        self.iterations.clear()

        for m in range(1, self.n_rounds + 1):
            # 1. Create and train a new weak learner on full data with sample_weight=w
            model = self.base_learner_factory()
            if m==1:
                    # Generate random indices for 10% of the training data
                    sample_size = int(0.1 * X_train.shape[0])
                    sampled_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)

                    # Select the subset of data
                    X_train_sample = X_train[sampled_indices]
                    y_train_sample = y_train[sampled_indices]

                    # Fit the model on the 10% subset for a weak learner
                    model.fit(X_train_sample, y_train_sample)
            else:
                model.fit(X_train, y_train,sample_weight=w)
                
               
           

            # 2. Compute weighted error
            y_pred_train = model.predict(X_train)
            misclassified = (y_pred_train != y_train).astype(float)
            err_m = np.sum(w * misclassified) / np.sum(w) + 1e-15

            # Early stop if 1 - error <= 1/K
            if 1 - err_m <= 1/self.n_classes + 1e-15:
                print(f"Stopping early at round {m}, weighted error = {err_m:.4f}")
                break

            # 3. Compute alpha_m
            #    alpha_m = ln((1 - err_m)/err_m) + ln(K-1)
            alpha_m = np.log((1 - err_m) / err_m) + np.log(self.n_classes - 1)

            # 4. Update sample weights
            #    w[i] *= exp(alpha_m) if y_pred != y_true
            #    then re-normalize
            w *= np.exp(alpha_m * misclassified)
            w /= np.sum(w)

            # Store the model and alpha
            self.alphas.append(alpha_m)
            self.models.append(model)
            self.iterations.append(m)

            # 5. Track metrics on training
            y_pred_train = self.predict(X_train)
            train_accuracy = np.mean(y_pred_train == y_train)
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(err_m)  # Weighted training error

            if X_valid is not None and y_valid is not None:
                y_pred_val = self.predict(X_valid)
                val_accuracy = np.mean(y_pred_val == y_valid)
                self.validation_accuracies.append(val_accuracy)
                val_misclassified = (y_pred_val != y_valid).astype(float)
                val_loss = np.mean(val_misclassified)
                self.validation_losses.append(val_loss)

                print(
                    f"Round {m}: Weighted Train Err={err_m:.4f}, "
                    f"Train Acc={train_accuracy:.4f}, "
                    f"Val Err={val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                )
            else:
                print(
                    f"Round {m}: Weighted Train Err={err_m:.4f}, "
                    f"Train Acc={train_accuracy:.4f}"
                )

    def predict(self, X,N=None):
        """
        Predict final class label using the boosted ensemble.
        """
        if not self.models:
            raise ValueError("No models trained. Call .fit() first.")
        scores = self._aggregate_scores(X,N)
        return np.argmax(scores, axis=1)

    def plot_metrics(self):
        """
        Plot training accuracy/error and optionally validation metrics.
        """
        if not self.train_accuracies or not self.train_losses:
            raise ValueError("No training metrics to plot. Did you call .fit()?")

        plt.figure(figsize=(12, 5))

        # 1) Accuracy subplot
        # plt.subplot(1, 2, 1)
        plt.plot(self.iterations, self.train_accuracies, marker="o", color="blue", label="Train Accuracy")
        if self.validation_accuracies:
            plt.plot(self.iterations, self.validation_accuracies, marker="s", color="green", label="Val Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation Accuracy")
        plt.grid(True)
        plt.legend()

        # 2) Weighted training error subplot
        # plt.subplot(1, 2, 2)
        # plt.plot(self.iterations, self.train_losses, marker="o", color="red", label="Weighted Train Error")
        # if self.validation_losses:
        #     plt.plot(self.iterations, self.validation_losses, marker="s", color="orange", label="Val Error")
        # plt.xlabel("Rounds")
        # plt.ylabel("Error")
        # plt.title("Training / Validation Error")
        # plt.grid(True)
        # plt.legend()

        plt.tight_layout()
        plt.show()
