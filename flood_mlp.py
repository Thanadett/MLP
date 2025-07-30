import copy
import math
import random
import csv


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.9, l2_reg=0.0):
        """Initialize the MLP model with given parameters"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg  # L2 regularization parameter

        self.layers = [input_size] + hidden_size + [output_size]
        self.num_layers = len(self.layers)

        self.weights = []
        self.biases = []
        self.prev_weights_deltas = []
        self.prev_biases_deltas = []

        self.activations = []
        self.z_values = []  # Store pre-activation values for better gradient computation

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
                limit = math.sqrt(6.0 / (input_size + output_size))
                weights = [
                    [random.uniform(-limit, limit) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            elif weight_init_method == 'he':
                limit = math.sqrt(2.0 / input_size)
                weights = [
                    [random.gauss(0, limit) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            elif weight_init_method == 'random':
                # Improved random initialization with smaller values
                limit = 0.1
                weights = [
                    [random.uniform(-limit, limit) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            else:
                raise ValueError(
                    f"Unsupported weight initialization method: {weight_init_method}")

            # Initialize biases to small random values
            biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

            self.weights.append(weights)
            self.biases.append(biases)

            self.prev_weights_deltas.append(
                [[0.0 for _ in range(input_size)] for _ in range(output_size)]
            )
            self.prev_biases_deltas.append([0.0 for _ in range(output_size)])

    def reset_weights(self, weight_init_method='xavier'):
        """Public method to reinitialize weights and biases."""
        self._initialize_weights(weight_init_method)

    def _sigmoid(self, x):
        """Sigmoid activation function with improved overflow handling"""
        x = max(-500, min(500, x))  # Clamp to prevent overflow
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _sigmoid_derivative(self, x):
        """Derivative of the sigmoid function"""
        return x * (1.0 - x)

    def _tanh(self, x):
        """Tanh activation function with improved overflow handling"""
        x = max(-500, min(500, x))  # Clamp to prevent overflow
        try:
            exp_2x = math.exp(2 * x)
            return (exp_2x - 1) / (exp_2x + 1)
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

    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function"""
        return max(alpha * x, x)

    def _leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of the Leaky ReLU function"""
        return 1.0 if x > 0 else alpha

    def _forward(self, inputs, activation_function='sigmoid'):
        """Forward pass through the network with improved gradient tracking"""
        self.activations = [inputs[:]]
        self.z_values = []  # Store pre-activation values
        current_inputs = inputs[:]

        for layer_idx in range(len(self.weights)):
            layer_outputs = []
            layer_z_values = []

            for neuron_idx in range(len(self.weights[layer_idx])):
                # Calculate weighted sum (pre-activation)
                weighted_sum = sum(
                    w * inp for w, inp in zip(self.weights[layer_idx][neuron_idx], current_inputs)
                )
                weighted_sum += self.biases[layer_idx][neuron_idx]
                layer_z_values.append(weighted_sum)

                # Apply activation function
                if activation_function == 'sigmoid':
                    output = self._sigmoid(weighted_sum)
                elif activation_function == 'tanh':
                    output = self._tanh(weighted_sum)
                elif activation_function == 'relu':
                    output = self._relu(weighted_sum)
                elif activation_function == 'leaky_relu':
                    output = self._leaky_relu(weighted_sum)
                else:
                    raise ValueError(
                        f"Unsupported activation function: {activation_function}")

                layer_outputs.append(output)

            self.z_values.append(layer_z_values[:])
            self.activations.append(layer_outputs[:])
            current_inputs = layer_outputs[:]

        return current_inputs

    def _backward(self, target, activation_function='sigmoid'):
        """Backward pass through the network with fixed gradient calculation"""
        # Calculate output layer error
        output_error = []
        for i in range(len(target)):
            error = target[i] - self.activations[-1][i]
            if activation_function == 'sigmoid':
                delta = error * \
                    self._sigmoid_derivative(self.activations[-1][i])
            elif activation_function == 'tanh':
                delta = error * self._tanh_derivative(self.activations[-1][i])
            elif activation_function == 'relu':
                delta = error * self._relu_derivative(self.z_values[-1][i])
            elif activation_function == 'leaky_relu':
                delta = error * \
                    self._leaky_relu_derivative(self.z_values[-1][i])
            else:
                raise ValueError(
                    f"Unsupported activation function: {activation_function}")
            output_error.append(delta)

        layer_errors = [output_error]

        # Backpropagate errors through hidden layers
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            current_error = []

            for neuron_idx in range(len(self.weights[layer_idx])):
                error = 0.0
                # Sum errors from next layer
                for next_neuron_idx in range(len(layer_errors[0])):
                    error += (layer_errors[0][next_neuron_idx] *
                              self.weights[layer_idx + 1][next_neuron_idx][neuron_idx])

                # Apply derivative of activation function
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
                            self.z_values[layer_idx][neuron_idx])
                elif activation_function == 'leaky_relu':
                    delta = error * \
                        self._leaky_relu_derivative(
                            self.z_values[layer_idx][neuron_idx])
                else:
                    raise ValueError(
                        f"Unsupported activation function: {activation_function}")

                current_error.append(delta)

            layer_errors.insert(0, current_error)

        # Update weights and biases with L2 regularization
        for layer_idx in range(len(self.weights)):
            for neuron_idx in range(len(self.weights[layer_idx])):
                for weight_idx in range(len(self.weights[layer_idx][neuron_idx])):
                    # Fixed: Use correct activation for input
                    input_to_use = self.activations[layer_idx][weight_idx]

                    # Calculate weight delta with L2 regularization
                    weight_delta = (
                        self.learning_rate *
                        layer_errors[layer_idx][neuron_idx] * input_to_use
                        - self.learning_rate * self.l2_reg *
                        self.weights[layer_idx][neuron_idx][weight_idx]
                        + self.momentum *
                        self.prev_weights_deltas[layer_idx][neuron_idx][weight_idx]
                    )

                    self.weights[layer_idx][neuron_idx][weight_idx] += weight_delta
                    self.prev_weights_deltas[layer_idx][neuron_idx][weight_idx] = weight_delta

                # Update bias
                bias_delta = (
                    self.learning_rate * layer_errors[layer_idx][neuron_idx]
                    + self.momentum *
                    self.prev_biases_deltas[layer_idx][neuron_idx]
                )

                self.biases[layer_idx][neuron_idx] += bias_delta
                self.prev_biases_deltas[layer_idx][neuron_idx] = bias_delta

    def train(self, X, y, epochs=1000, activation_function='sigmoid', verbose=True,
              X_val=None, y_val=None, patience=50, lr_decay=0.95, lr_decay_epochs=100):
        """Train the network with improved features"""
        mse_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        best_biases = None
        current_lr = self.learning_rate

        for epoch in range(epochs):
            total_error = 0.0

            # Shuffle training data for better convergence
            training_pairs = list(zip(X, y))
            random.shuffle(training_pairs)
            X_shuffled, y_shuffled = zip(*training_pairs)

            for i in range(len(X_shuffled)):
                outputs = self._forward(X_shuffled[i], activation_function)
                error = sum(
                    (target - outputs[j]) ** 2 for j, target in enumerate(y_shuffled[i]))
                total_error += error
                self._backward(y_shuffled[i], activation_function)

            mse = total_error / len(X)
            mse_history.append(mse)

            # Learning rate decay
            if epoch > 0 and epoch % lr_decay_epochs == 0:
                current_lr *= lr_decay
                self.learning_rate = current_lr

            # Validation
            if X_val is not None and y_val is not None:
                val_outputs = [self._forward(
                    x, activation_function) for x in X_val]
                val_loss = sum(
                    sum((target[j] - output[j]) **
                        2 for j in range(len(target)))
                    for target, output in zip(y_val, val_outputs)
                ) / len(X_val)

                val_loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = copy.deepcopy(self.weights)
                    best_biases = copy.deepcopy(self.biases)
                else:
                    epochs_no_improve += 1

                if verbose and epoch % 100 == 0:
                    print(
                        f"Epoch {epoch}, MSE: {mse:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

                if epochs_no_improve >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1} with best val loss: {best_val_loss:.6f}")
                    # Restore best weights
                    self.weights = best_weights
                    self.biases = best_biases
                    break
            else:
                if verbose and epoch % 100 == 0:
                    print(
                        f"Epoch {epoch}, MSE: {mse:.6f}, LR: {current_lr:.6f}")

        return mse_history, val_loss_history

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


def cross_validate_with_validation(X, y, k=10, val_split=0.2):
    """Perform k-fold cross-validation with validation split"""
    fold_size = len(X) // k
    folds = []

    indices = list(range(len(X)))
    random.shuffle(indices)

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(X)
        fold_indices = indices[start:end]

        train_indices = [idx for idx in indices if idx not in fold_indices]
        test_indices = fold_indices

        # Split training data into train and validation
        val_size = int(len(train_indices) * val_split)
        random.shuffle(train_indices)
        val_indices = train_indices[:val_size]
        actual_train_indices = train_indices[val_size:]

        X_train = [X[i] for i in actual_train_indices]
        y_train = [y[i] for i in actual_train_indices]
        X_val = [X[i] for i in val_indices]
        y_val = [y[i] for i in val_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        folds.append((X_train, y_train, X_val, y_val, X_test, y_test,
                     actual_train_indices, val_indices, test_indices))

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


def get_optimal_activation(weight_init):
    """Return the optimal activation function for the given weight initialization"""
    if weight_init == 'he':
        return 'relu'
    elif weight_init == 'xavier':
        return 'sigmoid'
    else:
        return 'sigmoid'  # Default for random and other methods


def run_experiment(X, y, hidden_layers, learning_rate, momentum,
                   activation='sigmoid', weight_init='xavier',
                   l2_reg=0.0, epochs=500, random_seed=None, verbose=True, show_folds=False):
    if random_seed is not None:
        random.seed(random_seed)

    # Automatically select activation function based on weight initialization
    activation = get_optimal_activation(weight_init)

    # Normalize features
    X_normalized = []
    normalization_params = []
    for feature_idx in range(len(X[0])):
        feature_values = [sample[feature_idx] for sample in X]
        normalized_values, min_val, max_val = normalize_data(feature_values)
        normalization_params.append((min_val, max_val))
        if feature_idx == 0:
            X_normalized = [[val] for val in normalized_values]
        else:
            for i, val in enumerate(normalized_values):
                X_normalized[i].append(val)

    # Normalize targets
    y_values = [sample[0] for sample in y]
    y_normalized, y_min, y_max = normalize_data(y_values)
    y_normalized = [[val] for val in y_normalized]

    folds = cross_validate_with_validation(X_normalized, y_normalized, k=10)
    cv_results = []

    if show_folds:
        print(f"\n{'='*80}")
        print(
            f"Hidden: {hidden_layers}, LR: {learning_rate}, Momentum: {momentum}")
        print(
            f"Activation: {activation}, Weight Init: {weight_init}")
        print(f"{'='*80}")
        print(f"{'Fold':<4} {'Train':<5} {'Val':<4} {'Test':<4} {'Train RMSE':<11} {'Val RMSE':<10} {'Test RMSE':<10} {'Test R²':<8} {'Epochs':<6}")
        print(f"{'-'*80}")

    for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices) in enumerate(folds):
        # Create and train model
        model = MLP(
            input_size=len(X_train[0]),
            hidden_size=hidden_layers,
            output_size=len(y_train[0]),
            learning_rate=learning_rate,
            momentum=momentum,
            l2_reg=l2_reg
        )
        model.reset_weights(weight_init_method=weight_init)

        # Train model with validation using the automatically selected activation
        mse_history, val_history = model.train(
            X_train, y_train, X_val=X_val, y_val=y_val, epochs=epochs,
            activation_function=activation, verbose=False, patience=50
        )

        # Calculate training performance
        train_predictions = []
        for train_sample in X_train:
            pred = model.predict(train_sample, activation_function=activation)
            train_predictions.append(pred)
        train_metrics = calculate_metrics(y_train, train_predictions)

        # Calculate validation performance
        val_predictions = []
        for val_sample in X_val:
            pred = model.predict(val_sample, activation_function=activation)
            val_predictions.append(pred)
        val_metrics = calculate_metrics(y_val, val_predictions)

        # Test model
        y_pred_normalized = []
        for test_sample in X_test:
            pred = model.predict(test_sample, activation_function=activation)
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
        metrics = calculate_metrics(y_test_denormalized, y_pred_denormalized)
        cv_results.append({
            'fold': fold_idx + 1,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_rmse': train_metrics['RMSE'],
            'val_rmse': val_metrics['RMSE'],
            'test_rmse': metrics['RMSE'],
            'test_r2': metrics['R2'],
            'test_mae': metrics['MAE'],
            'test_mse': metrics['MSE'],
            'epochs_trained': len(mse_history),
            'final_train_loss': mse_history[-1] if mse_history else 0,
            'final_val_loss': val_history[-1] if val_history else 0,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'activation_used': activation
        })

        if show_folds:
            print(f"{fold_idx+1:<4} {len(X_train):<5} {len(X_val):<4} {len(X_test):<4} "
                  f"{train_metrics['RMSE']:<11.4f} {val_metrics['RMSE']:<10.4f} "
                  f"{metrics['RMSE']:<10.4f} {metrics['R2']:<8.4f} {len(mse_history):<6}")

        if verbose:
            print(
                f"Fold {fold_idx + 1}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} | "
                f"Train RMSE={train_metrics['RMSE']:.4f}, Val RMSE={val_metrics['RMSE']:.4f}, "
                f"Test RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")

    if show_folds:
        print(f"{'-'*80}")

        # Calculate and show summary statistics
        train_rmse_values = [fold['train_rmse'] for fold in cv_results]
        val_rmse_values = [fold['val_rmse'] for fold in cv_results]
        test_rmse_values = [fold['test_rmse'] for fold in cv_results]
        test_r2_values = [fold['test_r2'] for fold in cv_results]

        print(f"{'Avg':<4} {'-':<5} {'-':<4} {'-':<4} "
              f"{sum(train_rmse_values)/len(train_rmse_values):<11.4f} "
              f"{sum(val_rmse_values)/len(val_rmse_values):<10.4f} "
              f"{sum(test_rmse_values)/len(test_rmse_values):<10.4f} "
              f"{sum(test_r2_values)/len(test_r2_values):<8.4f} {'-':<6}")

        print(f"{'='*80}")

    # Average metrics across folds
    avg_metrics = {}
    for metric in ['test_rmse', 'test_r2', 'test_mae', 'test_mse']:
        key = metric.replace('test_', '').upper()
        avg_metrics[key] = sum(fold[metric]
                               for fold in cv_results) / len(cv_results)

    return avg_metrics, cv_results


def main():
    """Main function to run systematic MLP experiments with automatic activation selection"""
    print("=== MLP Flood Prediction: Automatic Activation Selection Based on Weight Initialization ===\n")

    # Load data
    X, y = load_data('flood_data.csv')
    if X is None or y is None:
        return

    print(f"Loaded {len(X)} samples with {len(X[0])} features each")
    print("Using 10-fold cross-validation with automatic activation selection:")
    print("- Xavier initialization -> Sigmoid activation")
    print("- He initialization -> ReLU activation")
    print("- Random initialization -> Sigmoid activation\n")

    # Define smaller parameter ranges for demonstration
    hidden_architectures = [
        [5],
    ]

    learning_rates = [0.01]
    momentum_rates = [0.3]
    weight_inits = ['xavier']
    # hidden_architectures = [
    #     [5],
    #     [10],
    #     [8, 5],
    #     [12, 8],
    #     [10, 8, 5],
    #     [15, 12, 8],
    # ]

    # learning_rates = [0.01, 0.1, 0.2]
    # momentum_rates = [0.3, 0.5, 0.7, 0.9]
    # weight_inits = ['xavier', 'he']

    print("="*80)
    print("EXPERIMENT 1: HIDDEN NODES IMPACT WITH AUTO ACTIVATION SELECTION")
    print("="*80)

    hidden_results = []
    for i, hidden in enumerate(hidden_architectures):
        exp_name = f"Hidden-{'-'.join(map(str, hidden))}"
        print(
            f"\n--- Testing Architecture {i+1}/{len(hidden_architectures)}: {hidden} ---")

        # Test each architecture with different weight initializations
        for init_method in weight_inits:
            activation_used = get_optimal_activation(init_method)
            full_name = f"{exp_name}"
            print(f"\nTesting {full_name} with detailed 10-fold CV...")

            avg_metrics, cv_results = run_experiment(
                X, y, hidden, learning_rate=0.01, momentum=0.9,
                weight_init=init_method, l2_reg=0.0, epochs=500,
                random_seed=42, verbose=False, show_folds=True
            )

            hidden_results.append({
                'name': full_name,
                'avg_metrics': avg_metrics,
                'cv_results': cv_results,
                'hidden': hidden,
                'init': init_method,
                'activation': activation_used,
            })

            print(
                f"Overall Results: RMSE={avg_metrics['RMSE']:.4f}, R²={avg_metrics['R2']:.4f}" + '\n')

    print("\n" + "="*80)
    print("EXPERIMENT 2: LEARNING RATE IMPACT WITH DETAILED FOLD ANALYSIS")
    print("="*80)

    lr_results = []
    best_hidden = [8, 5]

    for i, lr in enumerate(learning_rates):
        exp_name = f"Learning_Rate-{lr}"
        print(
            f"\n--- Testing Learning Rate {i+1}/{len(learning_rates)}: {lr} ---")

        # Test each learning rate with different weight initializations
        for init_method in weight_inits:
            full_name = f"{exp_name} ({init_method})"
            print(f"\nTesting {full_name} with detailed 10-fold CV...")

            avg_metrics, cv_results = run_experiment(
                X, y, best_hidden, learning_rate=lr, momentum=0.9,
                activation='sigmoid', weight_init=init_method,
                l2_reg=0.0, epochs=500, random_seed=42, verbose=False, show_folds=True
            )

            lr_results.append({
                'name': full_name,
                'avg_metrics': avg_metrics,
                'lr': lr,
                'init': init_method,
            })
            print(
                f"Overall Results: RMSE={avg_metrics['RMSE']:.4f}, R²={avg_metrics['R2']:.4f}")

    print("\n" + "="*80)
    print("EXPERIMENT 3: MOMENTUM RATE IMPACT WITH DETAILED FOLD ANALYSIS")
    print("="*80)

    momentum_results = []
    best_lr = 0.01  # Use standard learning rate for momentum testing

    for i, momentum in enumerate(momentum_rates):
        exp_name = f"Momentum-{momentum}"
        print(
            f"\n--- Testing Momentum {i+1}/{len(momentum_rates)}: {momentum} ---")

        # Test each momentum with different weight initializations
        for init_method in weight_inits:
            full_name = f"{exp_name} ({init_method})"
            print(f"\nTesting {full_name} with detailed 10-fold CV...")

            avg_metrics, cv_results = run_experiment(
                X, y, best_hidden, learning_rate=best_lr, momentum=momentum,
                activation='sigmoid', weight_init=init_method,
                l2_reg=0.0, epochs=500, random_seed=42, verbose=False, show_folds=True
            )

            momentum_results.append({
                'name': full_name,
                'avg_metrics': avg_metrics,
                'momentum': momentum,
                'init': init_method,
            })
            print(
                f"Overall Results: RMSE={avg_metrics['RMSE']:.4f}, R²={avg_metrics['R2']:.4f}")

    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY (10-FOLD CROSS-VALIDATION)")
    print("="*80)

    print("\n1. HIDDEN ARCHITECTURE RESULTS:")
    print("-" * 80)
    print(f"{'Architecture':<20} {'Init Method':<10} {'RMSE':<15} {'R²':<15}")
    print("-" * 80)
    hidden_results.sort(key=lambda x: x['avg_metrics']['RMSE'])
    for result in hidden_results:
        arch_str = '-'.join(map(str, result['hidden']))
        print(f"{arch_str:<20} {result['init']:<10} "
              f"{result['avg_metrics']['RMSE']:<6.4f}"
              f"{result['avg_metrics']['R2']:<6.4f}")

    print("\n2. LEARNING RATE RESULTS:")
    print("-" * 80)
    print(f"{'Learning Rate':<15} {'Init Method':<10} {'RMSE':<15} {'R²':<15}")
    print("-" * 80)
    lr_results.sort(key=lambda x: x['avg_metrics']['RMSE'])
    for result in lr_results:
        print(f"{result['lr']:<15} {result['init']:<10} "
              f"{result['avg_metrics']['RMSE']:<6.4f} {result['avg_metrics']['R2']:<6.4f}")

    print("\n3. MOMENTUM RESULTS:")
    print("-" * 80)
    print(f"{'Momentum':<15} {'Init Method':<10} {'RMSE':<15} {'R²':<15}")
    print("-" * 80)
    momentum_results.sort(key=lambda x: x['avg_metrics']['RMSE'])
    for result in momentum_results:
        print(f"{result['momentum']:<15} {result['init']:<10} "
              f"{result['avg_metrics']['RMSE']:<6.4f}"
              f"{result['avg_metrics']['R2']:<6.4f}")

    # Find overall best result
    all_results = hidden_results + lr_results + momentum_results
    overall_best = min(all_results, key=lambda x: x['avg_metrics']['RMSE'])

    print("\n" + "="*110)
    print("OVERALL BEST CONFIGURATION :")
    print(f"Configuration: {overall_best['name']}")
    print(
        f"RMSE: {overall_best['avg_metrics']['RMSE']:.4f}")
    print(f"MAE: {overall_best['avg_metrics']['MAE']:.4f}")
    print(
        f"R²: {overall_best['avg_metrics']['R2']:.4f}")
    print("="*110)


if __name__ == "__main__":
    main()
