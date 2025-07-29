import csv
import random
import math
import copy


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.9):
        """Initialize the MLP model with given parameters"""

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.layers = [input_size] + hidden_size + [output_size]
        self.num_layers = len(self.layers)

        self.weights = []
        self.biases = []
        self.prev_weights_deltas = []
        self.prev_biases_deltas = []

        self.activations = []  # Initialize activations attribute

        self._initialize_weights()

    def _initialize_weights(self, weight_init_method='xavier'):
        """Initialize weights and biases based on the specified method"""
        self.weights = []
        self.biases = []
        self.prev_weights_deltas = []
        self.prev_biases_deltas = []

        for i in range(self.num_layers - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]

            if weight_init_method == 'xavier':
                limit = (6 / (input_size + output_size)) ** 0.5
                weights = [
                    [random.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)]
            elif weight_init_method == 'he':
                limit = (2 / input_size) ** 0.5
                weights = [
                    [random.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)]
            elif weight_init_method == 'random':
                weights = [
                    [random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
            else:
                raise ValueError(
                    f"Unsupported weight initialization method: {weight_init_method}")

            biases = [random.uniform(-1, 1) for _ in range(output_size)]

            self.weights.append(weights)
            self.biases.append(biases)

            self.prev_weights_deltas.append(
                [[0.0 for _ in range(input_size)] for _ in range(output_size)])
            self.prev_biases_deltas.append([0.0 for _ in range(output_size)])

    def reset_weights(self, weight_init_method='xavier'):
        """Public method to reinitialize weights and biases."""
        self._initialize_weights(weight_init_method)

    def _sigmoid(self, x):
        """Sigmoid activation function"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _sigmoid_derivative(self, x):
        """Derivative of the sigmoid function"""
        return x * (1.0 - x)

    def _tanh(self, x):
        """Tanh activation function"""
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else -1.0

    def _tanh_derivative(self, x):
        """Derivative of the tanh function"""
        return 1.0 - x * x

    def _relu(self, x):
        """ReLU activation function"""
        return max(0, x)

    def _relu_derivative(self, x):
        """Derivative of the ReLU function"""
        return 1.0 if x > 0 else 0.0

    def _forward(self, inputs, activation_function='sigmoid'):
        """Forward pass through the network"""
        self.activations = [inputs[:]]
        current_inputs = inputs[:]

        for layer_idx in range(len(self.weights)):
            layer_outputs = []

            for neuron_idx in range(len(self.weights[layer_idx])):

                weighted_sum = sum(
                    w * inp for w, inp in zip(self.weights[layer_idx][neuron_idx], current_inputs))
                weighted_sum += self.biases[layer_idx][neuron_idx]

                if activation_function == 'sigmoid':
                    output = self._sigmoid(weighted_sum)
                elif activation_function == 'tanh':
                    output = self._tanh(weighted_sum)
                elif activation_function == 'relu':
                    output = self._relu(weighted_sum)
                else:
                    raise ValueError(
                        f"Unsupported activation function: {activation_function}")

                layer_outputs.append(output)

            self.activations.append(layer_outputs[:])
            current_inputs = layer_outputs[:]

        return current_inputs

    def _backward(self, target, activation_function='sigmoid'):
        """Backward pass through the network"""
        output_error = []
        for i in range(len(target)):
            error = target[i] - self.activations[-1][i]
            if activation_function == 'sigmoid':
                delta = error * \
                    self._sigmoid_derivative(self.activations[-1][i])
            elif activation_function == 'tanh':
                delta = error * self._tanh_derivative(self.activations[-1][i])
            elif activation_function == 'relu':
                delta = error * self._relu_derivative(self.activations[-1][i])
            else:
                raise ValueError(
                    f"Unsupported activation function: {activation_function}")

            output_error.append(delta)

        layer_errors = [output_error]

        for layer_idx in range(len(self.weights) - 2, -1, -1):
            current_error = []

            for neuron_idx in range(len(self.weights[layer_idx])):
                error = 0.0

                for next_neuron_idx in range(len(layer_errors[0])):
                    error += layer_errors[0][next_neuron_idx] * \
                        self.weights[layer_idx +
                                     1][next_neuron_idx][neuron_idx]

                if activation_function == 'sigmoid':
                    delta = error * \
                        self._sigmoid_derivative(
                            self.activations[layer_idx + 1][neuron_idx])
                elif activation_function == 'tanh':
                    delta = error * \
                        self._tanh_derivative(
                            self.activations[layer_idx + 1][neuron_idx])
                elif activation_function == 'relu':
                    delta = error * \
                        self._relu_derivative(
                            self.activations[layer_idx + 1][neuron_idx])
                else:
                    raise ValueError(
                        f"Unsupported activation function: {activation_function}")

                current_error.append(delta)

            layer_errors.insert(0, current_error)

        for layer_idx in range(len(self.weights)):
            for neuron_idx in range(len(self.weights[layer_idx])):
                for weight_idx in range(len(self.weights[layer_idx][neuron_idx])):
                    if layer_idx == 0:
                        input_to_use = self.activations[0][weight_idx]
                    else:
                        input_to_use = self.activations[layer_idx][weight_idx]

                    weight_delta = (self.learning_rate * layer_errors[layer_idx][neuron_idx] * input_to_use
                                    + self.momentum * self.prev_weights_deltas[layer_idx][neuron_idx][weight_idx])

                    self.weights[layer_idx][neuron_idx][weight_idx] += weight_delta
                    self.prev_weights_deltas[layer_idx][neuron_idx][weight_idx] = weight_delta

                bias_delta = (self.learning_rate * layer_errors[layer_idx][neuron_idx]
                              + self.momentum * self.prev_biases_deltas[layer_idx][neuron_idx])

                self.biases[layer_idx][neuron_idx] += bias_delta
                self.prev_biases_deltas[layer_idx][neuron_idx] = bias_delta

    def train(self, X, y, epochs=1000, activation_function='sigmoid', verbose=True,
              X_val=None, y_val=None, patience=10):
        mse_history = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None

        for epoch in range(epochs):
            total_error = 0.0

            for i in range(len(X)):
                outputs = self._forward(X[i], activation_function)
                error = sum((target - outputs[j]) **
                            2 for j, target in enumerate(y[i]))
                total_error += error
                self._backward(y[i], activation_function)

            mse = total_error / len(X)
            mse_history.append(mse)

            if X_val is not None and y_val is not None:
                val_outputs = [self._forward(
                    x, activation_function) for x in X_val]
                val_loss = sum(
                    sum((target[j] - output[j]) **
                        2 for j in range(len(target)))
                    for target, output in zip(y_val, val_outputs)
                ) / len(X_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = copy.deepcopy(self.weights)
                else:
                    epochs_no_improve += 1

                if verbose and epoch % 100 == 0:
                    print(
                        f"Epoch {epoch}, MSE: {mse:.6f}, Val Loss: {val_loss:.6f}")

                if epochs_no_improve >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1} with best val loss: {best_val_loss:.6f}")
                    # คืนค่าน้ำหนักที่ดีที่สุด
                    self.weights = best_weights
                    break
            else:
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}, MSE: {mse:.6f}")

        return mse_history

    def predict(self, X, activation_function='sigmoid'):
        """Make predictions; auto-handle single or multiple inputs"""
        if isinstance(X[0], (int, float)):  # Single input vector
            return self._forward(X, activation_function)
        else:
            return [self._forward(x, activation_function) for x in X]


def load_data(file_path):
    """Load data from a CSV file and return features and targets"""
    X = []
    y = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                features = [
                    float(row["S1_t-3"]),
                    float(row["S1_t-2"]),
                    float(row["S1_t-1"]),
                    float(row["S1_t-0"]),
                    float(row["S2_t-3"]),
                    float(row["S2_t-2"]),
                    float(row["S2_t-1"]),
                    float(row["S2_t-0"]),
                ]
                target = [float(row["target"])]
                X.append(features)
                y.append(target)
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
        return None, None

    return X, y


def normalize_data(data):
    """Normalize the data to the range [0, 1]"""
    if len(data) == 0:
        return data, 0, 1

    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return [0.5] * len(data), min_val, max_val

    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val


def denormalize_data(normalized_data, min_val, max_val):
    """Denormalize the data from the range [0, 1] back to original scale"""
    if max_val == min_val:
        return [min_val] * len(normalized_data)

    denormalized = [x * (max_val - min_val) + min_val for x in normalized_data]
    return denormalized


def cross_validate(X, y, k=10):
    """Perform k-fold cross-validation"""
    fold_size = len(X) // k
    folds = []

    indices = list(range(len(X)))
    random.shuffle(indices)

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(X)
        fold_indices = indices[start:end]

        train_indices = [i for i in indices if i not in fold_indices]
        test_indices = fold_indices

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics for list of lists"""
    y_true_flat = [t[0] for t in y_true]
    y_pred_flat = [p[0] for p in y_pred]

    mse = sum((t - p) ** 2 for t, p in zip(y_true_flat,
              y_pred_flat)) / len(y_true_flat)

    rmse = math.sqrt(mse)

    mae = sum(abs(t - p)
              for t, p in zip(y_true_flat, y_pred_flat)) / len(y_true_flat)

    y_mean = sum(y_true_flat) / len(y_true_flat)
    ss_tot = sum((t - y_mean) ** 2 for t in y_true_flat)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true_flat, y_pred_flat))

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def run_experiment(X, y, hidden_layers, learning_rate, momentum, epochs=500, activation='sigmoid', weight_init='xavier', random_seed=None):
    """Run a single experiment with the given parameters"""
    if random_seed is not None:
        random.seed(random_seed)

    # Normalize features
    X_normalized = []
    normalized = []
    for feature_idx in range(len(X[0])):
        feature_values = [sample[feature_idx] for sample in X]
        normalized_values, min_val, max_val = normalize_data(feature_values)
        normalized.append((min_val, max_val))
        if feature_idx == 0:
            X_normalized = [[val] for val in normalized_values]
        else:
            for i, val in enumerate(normalized_values):
                X_normalized[i].append(val)

    # Normalize targets (y) - ทำแค่ครั้งเดียว
    y_values = [sample[0] for sample in y]
    y_normalized, y_min, y_max = normalize_data(y_values)
    y_normalized = [[val] for val in y_normalized]

    folds = cross_validate(X_normalized, y_normalized, k=10)

    cv_results = []

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
        # Create and train model
        model = MLP(input_size=len(X_train[0]), hidden_size=hidden_layers, output_size=len(y_train[0]),
                    learning_rate=learning_rate, momentum=momentum)
        model.reset_weights(weight_init_method=weight_init)

        # Train model
        model.train(
            X_train, y_train, epochs=epochs, activation_function=activation, verbose=False)

        # Test model
        y_pred_normalized = []
        for test_sample in X_test:
            pred = model.predict(
                test_sample, activation_function=activation)
            y_pred_normalized.append(pred)

        # Denormalize predictions
        y_pred_denormalized = []
        for pred in y_pred_normalized:
            denormalized_pred = denormalize_data(pred, y_min, y_max)
            y_pred_denormalized.append(denormalized_pred)

        y_test_denormalized = []
        for test_val in y_test:
            denormalized_test = denormalize_data(test_val, y_min, y_max)
            y_test_denormalized.append(denormalized_test)

        # Calculate metrics
        metrics = calculate_metrics(
            y_test_denormalized, y_pred_denormalized)
        cv_results.append(metrics)

        print(
            f"Fold {fold_idx + 1}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # Average metrics across folds
    avg_metrics = {}
    for metric in ['MSE', 'RMSE', 'MAE', 'R2']:
        avg_metrics[metric] = sum(fold[metric]
                                  for fold in cv_results) / len(cv_results)

    return avg_metrics, cv_results


def main():
    """Main function to run flood prediction experiments"""
    print("=== MLP Flood Prediction Experiments ===\n")

    # Load data
    X, y = load_data('flood_data.csv')
    if X is None or y is None:
        return

    print(f"Loaded {len(X)} samples with {len(X[0])} features each\n")

    experiments = [
        # Experiment 1: Different hidden layer configurations
        {"name": "Small Network", "hidden": [
            5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "relu"},
        {"name": "Medium Network", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "relu"},
        {"name": "Large Network", "hidden": [
            15, 10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "relu"},

        # Experiment 2: Different learning rates
        {"name": "Low Learning Rate", "hidden": [
            10, 5], "lr": 0.005, "momentum": 0.9, "init": "he", "activation": "relu"},
        {"name": "High Learning Rate", "hidden": [
            10, 5], "lr": 0.2, "momentum": 0.9, "init": "he", "activation": "relu"},


        # Experiment 3: Different momentum values
        {"name": "0.7 Momentum", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.7, "init": "he", "activation": "relu"},
        {"name": "0.8 Momentum", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.8, "init": "he", "activation": "relu"},
        {"name": "0.9 Momentum", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "relu"},

        # Experiment 4: Different weight initializations (explicit comparison)
        {"name": "Init: Xavier -> Activation: Sigmoid", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "xavier", "activation": "sigmoid"},
        {"name": "Init: He -> Activation: Sigmoid",     "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he",     "activation": "sigmoid"},
        {"name": "Init: Random -> Activation: Sigmoid", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "random", "activation": "sigmoid"},

        # Experiment 5: Different activation functions (init matched accordingly)
        {"name": "Init: Xavier -> Activation: Sigmoid", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "xavier", "activation": "sigmoid"},
        {"name": "Init: Xavier -> Activation: Tanh",    "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "xavier", "activation": "tanh"},
        {"name": "Init: Xavier -> Activation: ReLU", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "xavier", "activation": "relu"},
        {"name": "Init: He -> Activation: Sigmoid", "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "sigmoid"},
        {"name": "Init: He -> Activation: Tanh",    "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "tanh"},
        {"name": "Init: He -> Activation: ReLU",    "hidden": [
            10, 5], "lr": 0.01, "momentum": 0.9, "init": "he", "activation": "relu"},
    ]

    results = []

    for i, exp in enumerate(experiments):
        print(f"\n--- Running Experiment {i + 1}: {exp['name']} ---")
        print(f"Hidden Layers: {exp['hidden']}, Learning Rate: {exp['lr']}, Momentum: {exp['momentum']}, "
              f"Weight Init: {exp['init']}, Activation: {exp['activation']}", "\n")

        # Different seeds for reproducibility
        seeds = [42, 123, 456, 789, 101112]
        all_results = []

        for seed in seeds:
            print(f"Using random seed {seed}...")
            avg_metrics, cv_results = run_experiment(
                X, y, exp['hidden'], exp['lr'], exp['momentum'], activation=exp['activation'],
                weight_init=exp['init'], epochs=500, random_seed=seed
            )
            all_results.append(avg_metrics)

        # Average results different seeds
        final_metrics = {}
        for metric in ['MSE', 'RMSE', 'MAE', 'R2']:
            final_metrics[metric] = sum(result[metric]
                                        for result in all_results) / len(all_results)

        results.append((exp['name'], final_metrics, cv_results))

    # Print final results
    print(f"\nFinal Results for {exp['name']}:")
    print(f"RMSE: {final_metrics['RMSE']:.4f}")
    print(f"MAE: {final_metrics['MAE']:.4f}")
    print(f"R²: {final_metrics['R2']:.4f}")

    # Summary of all experiments
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Experiment':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-"*60)

    for name, metrics, _ in results:
        print(
            f"{name:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R2']:<10.4f}")

    # Find best experiment
    best_exp = min(results, key=lambda x: x[1]['RMSE'])
    print(f"\nBest performing experiment: {best_exp[0]}")
    print(f"RMSE: {best_exp[1]['RMSE']:.4f}")


if __name__ == "__main__":
    main()
# This is the main entry point for the script
# It runs the flood prediction experiments and prints the results
