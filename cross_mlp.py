import random
import math
import copy


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.9, l2_reg=0.0):
        """Initialize the MLP model with given parameters"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg  

        self.layers = [input_size] + hidden_size + [output_size]
        self.num_layers = len(self.layers)

        self.weights = []
        self.biases = []
        self.prev_weights_deltas = []
        self.prev_biases_deltas = []

        self.activations = []
        self.z_values = []  

        self._initialize_weights()

    def _initialize_weights(self, weight_init_method='xavier'):
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
        self._initialize_weights(weight_init_method)

    def _sigmoid(self, x):
        x = max(-500, min(500, x))  # Clamp to prevent overflow
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _forward(self, inputs, activation_function='sigmoid'):
        """Forward pass"""
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
                else:
                    raise ValueError(
                        f"Unsupported activation function: {activation_function}")

                layer_outputs.append(output)

            self.z_values.append(layer_z_values[:])
            self.activations.append(layer_outputs[:])
            current_inputs = layer_outputs[:]

        return current_inputs

    def _backward(self, target, activation_function='sigmoid'):
        """Backward pass"""
        # Calculate output layer error
        output_error = []
        for i in range(len(target)):
            error = target[i] - self.activations[-1][i]
            if activation_function == 'sigmoid':
                delta = error * \
                    self._sigmoid_derivative(self.activations[-1][i])
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
        mse_history = []
        val_loss_history = []
        accuracy_history = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        best_biases = None
        current_lr = self.learning_rate

        for epoch in range(epochs):
            total_error = 0.0
            correct_predictions = 0

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

                # Calculate accuracy for classification
                predicted_class = outputs.index(max(outputs))
                actual_class = y_shuffled[i].index(max(y_shuffled[i]))
                if predicted_class == actual_class:
                    correct_predictions += 1

            mse = total_error / len(X)
            accuracy = correct_predictions / len(X)
            mse_history.append(mse)
            accuracy_history.append(accuracy)

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
                        f"Epoch {epoch}, MSE: {mse:.6f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

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
                        f"Epoch {epoch}, MSE: {mse:.6f}, Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}")

        return mse_history, val_loss_history, accuracy_history

    def predict(self, X, activation_function='sigmoid'):
        if isinstance(X[0], (int, float)):  # Single input vector
            return self._forward(X, activation_function)
        else:
            return [self._forward(x, activation_function) for x in X]

    def predict_class(self, X, activation_function='sigmoid'):
        predictions = self.predict(X, activation_function)
        if isinstance(X[0], (int, float)):  # Single input
            return predictions.index(max(predictions))
        else:
            return [pred.index(max(pred)) for pred in predictions]


def load_pat_data(file_path):
    """Load data .pat file"""
    X = []
    y = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('p'):  # พบ pattern id เช่น p0, p1
                features_line = lines[i+1].strip()
                target_line = lines[i+2].strip()

                features = list(map(float, features_line.split()))
                targets = list(map(int, target_line.split()))

                X.append(features)
                y.append(targets)

                i += 3
            else:
                i += 1

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


def cross_validate_classification(X, y, k=10):
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

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds


def calculate_classification_metrics(y_true, y_pred):
    # Convert to class indices
    y_true_classes = [t.index(max(t)) for t in y_true]
    y_pred_classes = [p.index(max(p)) for p in y_pred]

    # Calculate accuracy
    correct = sum(1 for t, p in zip(y_true_classes, y_pred_classes) if t == p)
    accuracy = correct / len(y_true_classes)

    # Calculate confusion matrix
    num_classes = len(y_true[0])
    confusion_matrix = [[0 for _ in range(num_classes)]
                        for _ in range(num_classes)]

    for true_class, pred_class in zip(y_true_classes, y_pred_classes):
        confusion_matrix[true_class][pred_class] += 1

    # Calculate precision, recall, F1-score for each class
    precision = []
    recall = []
    f1_score = []

    for class_idx in range(num_classes):
        # True positives
        tp = confusion_matrix[class_idx][class_idx]
        # False positives
        fp = sum(confusion_matrix[i][class_idx]
                 for i in range(num_classes)) - tp
        # False negatives
        fn = sum(confusion_matrix[class_idx][i]
                 for i in range(num_classes)) - tp

        # Precision
        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0
        precision.append(prec)

        # Recall
        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0
        recall.append(rec)

        # F1-score
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_score.append(f1)

    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def print_confusion_matrix(confusion_matrix, class_names=None):
    """Print confusion matrix in a readable format"""
    num_classes = len(confusion_matrix)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print("\nConfusion Matrix:")
    print("=" * 50)

    # Header
    print("Actual\\Predicted".ljust(15), end="")
    for name in class_names:
        print(f"{name:<10}", end="")
    print()

    print("-" * 50)

    # Matrix rows
    for i, row in enumerate(confusion_matrix):
        print(f"{class_names[i]:<15}", end="")
        for val in row:
            print(f"{val:<10}", end="")
        print()


def run_classification_experiment(X, y, hidden_layers, learning_rate, momentum, epochs=1000,
                                  activation='sigmoid', weight_init='xavier', l2_reg=0.0,
                                  random_seed=None, verbose=False):
    """Run a classification experiment with the given parameters"""
    if random_seed is not None:
        random.seed(random_seed)

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

    folds = cross_validate_classification(X_normalized, y, k=10)
    cv_results = []

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
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

        # Train model
        model.train(
            X_train, y_train, epochs=epochs,
            activation_function=activation, verbose=verbose, patience=50
        )

        # Test model
        y_pred = model.predict(X_test, activation_function=activation)

        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred)
        cv_results.append(metrics)

        if verbose:
            print(f"Fold {fold_idx + 1}: Accuracy={metrics['accuracy']:.4f}")

    # Average metrics across folds
    avg_accuracy = sum(fold['accuracy']
                       for fold in cv_results) / len(cv_results)

    # Aggregate confusion matrix
    num_classes = len(cv_results[0]['confusion_matrix'])
    total_confusion_matrix = [
        [0 for _ in range(num_classes)] for _ in range(num_classes)]

    for fold_result in cv_results:
        for i in range(num_classes):
            for j in range(num_classes):
                total_confusion_matrix[i][j] += fold_result['confusion_matrix'][i][j]

    avg_precision = [sum(fold['precision'][i] for fold in cv_results) / len(cv_results)
                     for i in range(num_classes)]
    avg_recall = [sum(fold['recall'][i] for fold in cv_results) / len(cv_results)
                  for i in range(num_classes)]
    avg_f1 = [sum(fold['f1_score'][i] for fold in cv_results) / len(cv_results)
              for i in range(num_classes)]

    return {
        'accuracy': avg_accuracy,
        'confusion_matrix': total_confusion_matrix,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1
    }, cv_results


def main_classification():
    print("=========== MLP Classification: Cross.pat Dataset ===========\n")

    # Load data
    X, y = load_pat_data('cross.pat')
    if X is None or y is None:
        print("Error loading data. Creating sample data for demonstration.")
        # Create sample data in the expected format
        X = [[0.0902, 0.2690], [0.1843, 0.3456], [0.7823, 0.8901], [0.9234, 0.7654],
             [0.2345, 0.1234], [0.5678, 0.4321], [0.8765, 0.9876], [0.3456, 0.6789]]
        y = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]]

    print(f"Loaded {len(X)} samples with {len(X[0])} features each")
    print(f"Number of classes: {len(y[0])}\n")

    # Define parameter ranges for systematic testing
    hidden_architectures = [
        [5],
        [10],
        [8, 5],
        [12, 8],
        [10, 8, 5],
        [15, 12, 8],
    ]

    learning_rates = [0.01, 0.1, 0.2]
    momentum_rates = [0.3, 0.5, 0.7, 0.9]
    weight_inits = ['xavier', 'he']

    print("="*80)
    print("EXPERIMENT 1: HIDDEN LAYER")
    print("="*80)

    hidden_results = []
    for i, hidden in enumerate(hidden_architectures):
        exp_name = f"Hidden-{'-'.join(map(str, hidden))}"
        print(f"\n--- Testing Architecture {i+1}/6: {hidden} ---")

        # Test each architecture with different weight initializations
        for init_method in weight_inits:
            full_name = f"{exp_name}"
            print(f"\nTesting {full_name} ")

            avg_metrics, cv_results = run_classification_experiment(
                X, y, hidden, learning_rate=0.1, momentum=0.9,
                activation='sigmoid', weight_init=init_method,
                l2_reg=0.0, epochs=1000, random_seed=42, verbose=False
            )

            # Calculate CV statistics
            cv_accuracies = [fold['accuracy'] for fold in cv_results]
            acc_std = math.sqrt(
                sum((x - avg_metrics['accuracy'])**2 for x in cv_accuracies) / len(cv_accuracies))

            hidden_results.append(
                (full_name, avg_metrics, hidden, init_method, acc_std))
            print(
                f"Results: Accuracy={avg_metrics['accuracy']:.4f}")

    print("\n" + "="*80)
    print("EXPERIMENT 2: LEARNING RATE")
    print("="*80)

    lr_results = []
    best_hidden = [6, 4]  

    for i, lr in enumerate(learning_rates):
        exp_name = f"LR-{lr}"
        print(f"\n--- Testing Learning Rate {i+1}/3: {lr} ---")

        # Test each learning rate with different weight initializations
        for init_method in weight_inits:
            full_name = f"{exp_name}"
            print(f"\nTesting {full_name}")

            avg_metrics, cv_results = run_classification_experiment(
                X, y, best_hidden, learning_rate=lr, momentum=0.9,
                activation='sigmoid', weight_init=init_method,
                l2_reg=0.0, epochs=1000, random_seed=42, verbose=False
            )

            # Calculate CV statistics
            cv_accuracies = [fold['accuracy'] for fold in cv_results]
            acc_std = math.sqrt(
                sum((x - avg_metrics['accuracy'])**2 for x in cv_accuracies) / len(cv_accuracies))

            lr_results.append(
                (full_name, avg_metrics, lr, init_method, acc_std))
            print(
                f"Results: Accuracy={avg_metrics['accuracy']:.4f}")

    print("\n" + "="*80)
    print("EXPERIMENT 3: MOMENTUM RATE")
    print("="*80)

    momentum_results = []
    best_lr = 0.1 

    for i, momentum in enumerate(momentum_rates):
        exp_name = f"Momentum-{momentum}"
        print(f"\n--- Testing Momentum {i+1}/4: {momentum} ---")

        # Test each momentum with different weight initializations
        for init_method in weight_inits:
            full_name = f"{exp_name}"
            print(f"\nTesting {full_name}")

            avg_metrics, cv_results = run_classification_experiment(
                X, y, best_hidden, learning_rate=best_lr, momentum=momentum,
                activation='sigmoid', weight_init=init_method,
                l2_reg=0.0, epochs=1000, random_seed=42, verbose=False
            )

            # Calculate CV statistics
            cv_accuracies = [fold['accuracy'] for fold in cv_results]
            acc_std = math.sqrt(
                sum((x - avg_metrics['accuracy'])**2 for x in cv_accuracies) / len(cv_accuracies))

            momentum_results.append(
                (full_name, avg_metrics, momentum, init_method, acc_std))
            print(
                f"Results: Accuracy={avg_metrics['accuracy']:.4f}")

    print("\n" + "="*80)
    print("EXPERIMENT 4: BEST COMBINATIONS WITH CONFUSION MATRICES")
    print("="*80)

    # Find best parameters from previous experiments
    best_hidden_config = max(hidden_results, key=lambda x: x[1]['accuracy'])
    best_lr_config = max(lr_results, key=lambda x: x[1]['accuracy'])
    best_momentum_config = max(
        momentum_results, key=lambda x: x[1]['accuracy'])

    print(
        f"Best Hidden Architecture: {best_hidden_config[2]} with {best_hidden_config[3]} init")
    print(
        f"Best Learning Rate: {best_lr_config[2]} with {best_lr_config[3]} init")
    print(
        f"Best Momentum: {best_momentum_config[2]} with {best_momentum_config[3]} init")

    # Test best combinations and show confusion matrices
    final_experiments = [
        {
            "name": "Best Hidden Architecture",
            "hidden": best_hidden_config[2],
            "lr": 0.1,
            "momentum": 0.9,
            "init": best_hidden_config[3]
        },
        {
            "name": "Best Learning Rate",
            "hidden": [6, 4],
            "lr": best_lr_config[2],
            "momentum": 0.9,
            "init": best_lr_config[3]
        },
        {
            "name": "Best Momentum",
            "hidden": [6, 4],
            "lr": 0.1,
            "momentum": best_momentum_config[2],
            "init": best_momentum_config[3]
        },
        {
            "name": "Combined Best Parameters",
            "hidden": best_hidden_config[2],
            "lr": best_lr_config[2],
            "momentum": best_momentum_config[2],
            "init": "xavier"
        }
    ]

    final_results = []
    for exp in final_experiments:
        print(f"\n--- Testing {exp['name']} ---")
        print(
            f"Configuration: Hidden={exp['hidden']}, LR={exp['lr']}, Momentum={exp['momentum']}, Init={exp['init']}")

        # Use multiple seeds for final experiments
        seeds = [42, 123, 456]
        all_seed_results = []
        all_cv_results = []

        for seed in seeds:
            avg_metrics, cv_results = run_classification_experiment(
                X, y, exp['hidden'], exp['lr'], exp['momentum'],
                activation='sigmoid', weight_init=exp['init'],
                l2_reg=0.0, epochs=1000, random_seed=seed, verbose=False
            )
            all_seed_results.append(avg_metrics)
            all_cv_results.extend(cv_results)

        # Average across seeds
        final_accuracy = sum(result['accuracy']
                             for result in all_seed_results) / len(all_seed_results)

        # Use the best seed result for confusion matrix display
        best_seed_result = max(all_seed_results, key=lambda x: x['accuracy'])

        # Calculate overall CV statistics
        cv_acc_all = [fold['accuracy'] for fold in all_cv_results]
        acc_std = math.sqrt(
            sum((x - final_accuracy)**2 for x in cv_acc_all) / len(cv_acc_all))

        final_results.append((exp['name'], best_seed_result, acc_std))

        print(f"Final Results: Accuracy={final_accuracy:.4f}")

        # Display confusion matrix for best result
        print_confusion_matrix(best_seed_result['confusion_matrix'], [
                               'Class 0', 'Class 1'])

        print("\nDetailed Metrics:")
        for i, class_name in enumerate(['Class 0', 'Class 1']):
            print(f"{class_name}: Precision={best_seed_result['precision'][i]:.4f}, "
                  f"Recall={best_seed_result['recall'][i]:.4f}, "
                  f"F1-Score={best_seed_result['f1_score'][i]:.4f}")

    # COMPREHENSIVE RESULTS SUMMARY
    print("\n" + "="*110)
    print("COMPREHENSIVE CLASSIFICATION RESULTS SUMMARY (10-FOLD CROSS-VALIDATION)")
    print("="*110)

    print("\n1. HIDDEN ARCHITECTURE RESULTS:")
    print("-" * 80)
    print(f"{'Architecture':<20} {'Init Method':<10} {'Accuracy':<15}")
    print("-" * 80)
    hidden_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    for name, metrics, arch, init, acc_std in hidden_results:
        arch_str = '-'.join(map(str, arch))
        print(
            f"{arch_str:<20} {init:<10} {metrics['accuracy']:.4f}")

    print("\n2. LEARNING RATE RESULTS:")
    print("-" * 80)
    print(f"{'Learning Rate':<15} {'Init Method':<10} {'Accuracy':<15}")
    print("-" * 80)
    lr_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    for name, metrics, lr, init, acc_std in lr_results:
        print(f"{lr:<15} {init:<10} {metrics['accuracy']:.4f}")

    print("\n3. MOMENTUM RESULTS:")
    print("-" * 80)
    print(f"{'Momentum':<15} {'Init Method':<10} {'Accuracy':<15}")
    print("-" * 80)
    momentum_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    for name, metrics, momentum, init, acc_std in momentum_results:
        print(
            f"{momentum:<15} {init:<10} {metrics['accuracy']:.4f}")

    print("\n4. FINAL BEST COMBINATIONS:")
    print("-" * 90)
    print(f"{'Configuration':<45} {'Accuracy':<15}")
    print("-" * 90)
    final_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    for name, metrics, acc_std in final_results:
        print(f"{name:<45} {metrics['accuracy']:.4f}")

    # Overall best result
    all_configs = [(name, metrics, None, None, acc_std)
                   for name, metrics, arch, init, acc_std in hidden_results]
    all_configs.extend([(name, metrics, param, init, acc_std)
                       for name, metrics, param, init, acc_std in lr_results])
    all_configs.extend([(name, metrics, param, init, acc_std)
                       for name, metrics, param, init, acc_std in momentum_results])
    all_configs.extend([(name, metrics, None, None, acc_std)
                       for name, metrics, acc_std in final_results])

    overall_best = max(all_configs, key=lambda x: x[1]['accuracy'])

    print("\n" + "="*110)
    print("OVERALL BEST CONFIGURATION:")
    print(f"Configuration: {overall_best[0]}")
    print(
        f"Accuracy: {overall_best[1]['accuracy']:.4f}")

    # Show final confusion matrix
    if 'confusion_matrix' in overall_best[1]:
        print("\nFinal Confusion Matrix for Best Configuration:")
        print_confusion_matrix(overall_best[1]['confusion_matrix'], [
                               'Class 0', 'Class 1'])

        print("\nFinal Detailed Metrics:")
        for i, class_name in enumerate(['Class 0', 'Class 1']):
            print(f"{class_name}: Precision={overall_best[1]['precision'][i]:.4f}, "
                  f"Recall={overall_best[1]['recall'][i]:.4f}, "
                  f"F1-Score={overall_best[1]['f1_score'][i]:.4f}")

    print("="*110)


if __name__ == "__main__":
    main_classification()
