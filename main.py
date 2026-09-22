from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, replace
from itertools import product
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Vector = np.ndarray
Matrix = np.ndarray
ParameterPairs = list[tuple[np.ndarray, np.ndarray]]

SEED = 54
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
DEPTHS = (3, 5, 7)
WIDTHS = (13, 26)
LEARNING_RATES = (0.003, 0.01)
GAMMA_SCALES = (0.5, 2.0)
MAX_EPOCHS = 250
PATIENCE = 40
BATCH_SIZE = 32
REPEAT_SEEDS = (71, 72, 73)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "wine.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


@dataclass
class Dataset:
    x: Matrix
    y: Vector
    feature_names: list[str]
    label_names: list[str]


@dataclass
class TrainTestSplit:
    x_train: Matrix
    x_test: Matrix
    y_train: Vector
    y_test: Vector


@dataclass
class Preprocessor:
    median: Vector
    mean: Vector
    std: Vector

    def transform(self, x: Matrix) -> Matrix:
        if x.ndim != 2 or x.shape[1] != len(self.mean) or np.any(np.isinf(x)):
            raise ValueError("Input must match fitted features and contain no infinities.")
        x_filled = np.where(np.isnan(x), self.median, x)
        return (x_filled - self.mean) / self.std


def activation_forward(x: Matrix, name: str) -> Matrix:
    if name == "identity":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown activation: {name}")


def activation_derivative(x: Matrix, name: str) -> Matrix:
    if name == "identity":
        return np.ones_like(x)
    if name == "relu":
        return (x > 0.0).astype(float)
    if name == "tanh":
        return 1.0 - np.tanh(x) ** 2
    raise ValueError(f"Unknown activation: {name}")


class DenseLayer:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: str,
        rng: np.random.Generator,
    ) -> None:
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("Layer dimensions must be positive.")
        if activation not in ("identity", "relu", "tanh"):
            raise ValueError(f"Unknown activation: {activation}")

        scale = (
            np.sqrt(2.0 / n_inputs)
            if activation == "relu" else np.sqrt(2.0 / (n_inputs + n_outputs))
        )
        self.weights = rng.normal(0.0, scale, size=(n_inputs, n_outputs))
        self.bias = np.zeros(n_outputs)
        self.activation = activation
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.input_cache: Matrix | None = None
        self.preactivation_cache: Matrix | None = None

    def forward(self, x: Matrix) -> Matrix:
        self.input_cache = None
        self.preactivation_cache = None
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0]:
            raise ValueError("Input must have shape (n_samples, n_inputs).")

        self.input_cache = x.copy()
        self.preactivation_cache = self.input_cache @ self.weights + self.bias
        return activation_forward(self.preactivation_cache, self.activation).copy()

    def backward(self, grad_output: Matrix) -> Matrix:
        if self.input_cache is None or self.preactivation_cache is None:
            raise RuntimeError("Call forward before backward.")
        if grad_output.shape != self.preactivation_cache.shape:
            raise ValueError("Output gradient must have the same shape as the layer output.")

        grad_z = grad_output * activation_derivative(self.preactivation_cache, self.activation)
        self.grad_weights[...] = self.input_cache.T @ grad_z
        self.grad_bias[...] = np.sum(grad_z, axis=0)
        return grad_z @ self.weights.T

    def parameters_and_gradients(self) -> ParameterPairs:
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]


class RBFLayer:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        rng: np.random.Generator,
        gamma: float | None = None,
    ) -> None:
        if n_inputs < 1 or n_outputs < 1:
            raise ValueError("Layer dimensions must be positive.")
        if gamma is None:
            gamma = 1.0 / n_inputs
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive.")

        self.centers = rng.normal(0.0, 1.0, size=(n_outputs, n_inputs))
        self.gamma = float(gamma)
        self.grad_centers = np.zeros_like(self.centers)
        self.difference_cache: np.ndarray | None = None
        self.output_cache: Matrix | None = None

    def forward(self, x: Matrix) -> Matrix:
        self.difference_cache = None
        self.output_cache = None
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1]:
            raise ValueError("Input must have shape (n_samples, n_inputs).")

        self.difference_cache = x[:, None, :] - self.centers[None, :, :]
        squared_distances = np.sum(self.difference_cache ** 2, axis=2)
        self.output_cache = np.exp(-self.gamma * squared_distances)
        return self.output_cache.copy()

    def backward(self, grad_output: Matrix) -> Matrix:
        if self.difference_cache is None or self.output_cache is None:
            raise RuntimeError("Call forward before backward.")
        if grad_output.shape != self.output_cache.shape:
            raise ValueError("Output gradient must have the same shape as the layer output.")

        grad_pairs = (
            -2.0 * self.gamma
            * (grad_output * self.output_cache)[:, :, None]
            * self.difference_cache
        )
        self.grad_centers[...] = -np.sum(grad_pairs, axis=0)
        return np.sum(grad_pairs, axis=1)

    def parameters_and_gradients(self) -> ParameterPairs:
        return [(self.centers, self.grad_centers)]


class ResidualLayer:
    def __init__(self, layer: DenseLayer | RBFLayer | ResidualLayer) -> None:
        self.layer = layer
        self.output_shape: tuple[int, int] | None = None

    def forward(self, x: Matrix) -> Matrix:
        self.output_shape = None
        output = self.layer.forward(x)
        if output.shape != x.shape:
            raise ValueError("Residual connection requires matching input and output shapes.")

        self.output_shape = output.shape
        return x + output

    def backward(self, grad_output: Matrix) -> Matrix:
        if self.output_shape is None:
            raise RuntimeError("Call a successful forward before backward.")
        if grad_output.shape != self.output_shape:
            raise ValueError("Output gradient must have the same shape as the layer output.")

        return grad_output + self.layer.backward(grad_output)

    def parameters_and_gradients(self) -> ParameterPairs:
        return self.layer.parameters_and_gradients()


class SoftArgMaxCrossEntropy:
    def __init__(self) -> None:
        self.gradient_cache: Matrix | None = None

    def forward(self, logits: Matrix, targets: np.ndarray) -> float:
        self.gradient_cache = None
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
            raise ValueError("Logits must have shape (n_samples, n_classes), with at least two classes.")
        if not np.all(np.isfinite(logits)):
            raise ValueError("Logits must be finite.")
        n_samples, n_classes = logits.shape
        if targets.ndim == 1:
            if targets.shape != (n_samples,) or not np.issubdtype(targets.dtype, np.integer):
                raise ValueError("Class labels must be an integer vector matching the batch.")
            if np.any(targets < 0) or np.any(targets >= n_classes):
                raise ValueError("Class labels must be in [0, n_classes).")
            distribution = np.eye(n_classes)[targets]
        elif targets.shape == logits.shape:
            if not np.all(np.isfinite(targets)) or np.any(targets < 0.0):
                raise ValueError("Target probabilities must be finite and nonnegative.")
            if not np.allclose(targets.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError("Each target distribution must sum to one.")
            distribution = targets / targets.sum(axis=1, keepdims=True)
        else:
            raise ValueError("Targets must be class labels or a matrix of target distributions.")

        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probabilities = np.exp(log_probabilities)
        self.gradient_cache = (probabilities - distribution) / n_samples
        return float(-np.sum(distribution * log_probabilities) / n_samples)

    def backward(self) -> Matrix:
        if self.gradient_cache is None:
            raise RuntimeError("Call a successful forward before backward.")
        return self.gradient_cache.copy()


class Sequential:
    def __init__(self, layers: list[DenseLayer | RBFLayer | ResidualLayer]) -> None:
        if not layers:
            raise ValueError("A model must contain at least one layer.")
        self.layers = layers

    def forward(self, x: Matrix) -> Matrix:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient: Matrix) -> Matrix:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters_and_gradients(self) -> ParameterPairs:
        return [pair for layer in self.layers for pair in layer.parameters_and_gradients()]

    def parameter_count(self) -> int:
        return sum(parameter.size for parameter, _ in self.parameters_and_gradients())


class Adam:
    def __init__(
        self,
        parameters: ParameterPairs,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if not parameters:
            raise ValueError("At least one parameter is required.")

        seen = set()
        for parameter, gradient in parameters:
            if parameter.shape != gradient.shape:
                raise ValueError("Parameter and gradient shapes must match.")
            if not all(np.issubdtype(array.dtype, np.floating) for array in (parameter, gradient)):
                raise ValueError("Parameters and gradients must be floating-point arrays.")
            if id(parameter) in seen:
                raise ValueError("Each parameter must be passed only once.")
            seen.add(id(parameter))

        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_steps = 0
        self.first_moments = [np.zeros_like(parameter) for parameter, _ in parameters]
        self.second_moments = [np.zeros_like(parameter) for parameter, _ in parameters]

    def step(self) -> None:
        for parameter, gradient in self.parameters:
            if parameter.shape != gradient.shape or not np.all(np.isfinite(gradient)):
                raise ValueError("Gradients must be finite and match parameter shapes.")

        next_step = self.n_steps + 1
        correction1 = 1.0 - self.beta1 ** next_step
        correction2 = 1.0 - self.beta2 ** next_step
        for (parameter, gradient), first, second in zip(
            self.parameters, self.first_moments, self.second_moments,
        ):
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                next_first = self.beta1 * first + (1.0 - self.beta1) * gradient
                next_second = self.beta2 * second + (1.0 - self.beta2) * gradient ** 2
                change = self.learning_rate * (next_first / correction1) / (
                    np.sqrt(next_second / correction2) + self.epsilon
                )
                next_parameter = parameter - change
            if not np.all(np.isfinite(next_parameter)):
                raise FloatingPointError("Adam produced nonfinite parameters.")
            parameter[...] = next_parameter
            first[...] = next_first
            second[...] = next_second
        self.n_steps = next_step


def load_wine_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        if len(header) < 2 or header[0] != "class" or len(set(header)) != len(header):
            raise ValueError("CSV requires a class column and unique feature names.")
        rows = [row for row in reader if row]
    if not rows or any(len(row) != len(header) or not row[0].strip() for row in rows):
        raise ValueError("CSV rows must match the header and have nonempty class labels.")

    feature_names = header[1:]
    x = np.array(
        [[float(value) if value.strip() else np.nan for value in row[1:]] for row in rows],
        dtype=float,
    )
    label_names, y = np.unique([row[0] for row in rows], return_inverse=True)
    return Dataset(
        x=x,
        y=y,
        feature_names=feature_names,
        label_names=label_names.tolist(),
    )


def stratified_split_indices(y: Vector, test_size: float, seed: int) -> tuple[Vector, Vector]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if len(y) == 0:
        raise ValueError("At least one class is required.")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) < 2:
            raise ValueError("Each class must contain at least two samples.")

        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, int(round(test_size * len(indices)))))
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def split_dataset(dataset: Dataset, test_size: float, seed: int) -> TrainTestSplit:
    train_idx, test_idx = stratified_split_indices(dataset.y, test_size, seed)
    return TrainTestSplit(
        x_train=dataset.x[train_idx],
        x_test=dataset.x[test_idx],
        y_train=dataset.y[train_idx],
        y_test=dataset.y[test_idx],
    )


def fit_preprocessor(x_train: Matrix) -> Preprocessor:
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError("Training data must be a nonempty matrix.")
    if x_train.shape[1] == 0 or np.any(np.isinf(x_train)):
        raise ValueError("Training features must be nonempty and contain no infinities.")
    if np.any(np.all(np.isnan(x_train), axis=0)):
        raise ValueError("Each feature must have at least one observed training value.")

    median = np.nanmedian(x_train, axis=0)
    x_filled = np.where(np.isnan(x_train), median, x_train)
    mean = np.mean(x_filled, axis=0)
    std = np.std(x_filled, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return Preprocessor(median=median, mean=mean, std=std)


@dataclass(frozen=True)
class ModelConfig:
    family: str
    depth: int
    width: int
    residual: bool
    learning_rate: float
    activation: str = "tanh"
    gamma_scale: float = 1.0


@dataclass
class TrainingResult:
    model: Sequential
    history: list[dict]
    best_epoch: int
    best_val_loss: float


@dataclass
class SelectedModel:
    config: ModelConfig
    training: TrainingResult
    val_accuracy: float
    val_f1: float


def classification_metrics(y_true: Vector, logits: Matrix) -> tuple[float, float]:
    prediction = logits.argmax(axis=1)
    accuracy = float(np.mean(prediction == y_true))
    scores = []
    for label in range(logits.shape[1]):
        tp = np.sum((prediction == label) & (y_true == label))
        fp = np.sum((prediction == label) & (y_true != label))
        fn = np.sum((prediction != label) & (y_true == label))
        denominator = 2 * tp + fp + fn
        scores.append(float(2 * tp / denominator) if denominator else 0.0)
    return accuracy, float(np.mean(scores))


def build_model(config: ModelConfig, x_fit: Matrix, y_fit: Vector, seed: int) -> Sequential:
    if config.family not in ("dense", "rbf", "mixed") or config.depth < 3 or config.width < 1:
        raise ValueError("Experiments require a known family, depth >= 3 and positive width.")
    classes = np.unique(y_fit)
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Training labels must cover consecutive classes starting at zero.")
    layers = []
    representation = x_fit
    rng = np.random.default_rng(seed)
    for index in range(config.depth):
        is_last = index == config.depth - 1
        n_inputs = representation.shape[1]
        n_outputs = len(classes) if is_last else config.width
        use_rbf = config.family == "rbf" or (config.family == "mixed" and index % 2 == 1 and not is_last)
        if use_rbf:
            distance_scale = max(2.0 * np.var(representation, axis=0).sum(), 1e-3)
            layer = RBFLayer(n_inputs, n_outputs, rng, gamma=config.gamma_scale / distance_scale)
            if is_last:
                layer.centers[:] = [representation[y_fit == label].mean(axis=0) for label in classes]
            else:
                indices = rng.choice(len(representation), n_outputs, replace=n_outputs > len(representation))
                layer.centers[:] = representation[indices]
        else:
            layer = DenseLayer(n_inputs, n_outputs, "identity" if is_last else config.activation, rng)
        # Тут так, потому что остаточные слои можем только внутрь пихнуть
        if config.residual and 0 < index < config.depth - 1:
            layer = ResidualLayer(layer)
        layers.append(layer)
        representation = layer.forward(representation)
    return Sequential(layers)


def fit_model(
    model: Sequential,
    x_train: Matrix,
    y_train: Vector,
    learning_rate: float,
    seed: int,
    epochs: int,
    validation: tuple[Matrix, Vector] | None = None,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> TrainingResult:
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Epochs, patience and batch size must be positive.")
    optimizer = Adam(model.parameters_and_gradients(), learning_rate=learning_rate)
    loss_fn = SoftArgMaxCrossEntropy()
    rng = np.random.default_rng(seed + 1000)
    history = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_parameters = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = model.forward(x_train[indices])
            loss_fn.forward(logits, y_train[indices])
            model.backward(loss_fn.backward())
            optimizer.step()
        train_logits = model.forward(x_train)
        train_loss = loss_fn.forward(train_logits, y_train)
        train_accuracy, train_f1 = classification_metrics(y_train, train_logits)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "train_f1": train_f1}
        if validation is not None:
            x_val, y_val = validation
            val_logits = model.forward(x_val)
            val_loss = loss_fn.forward(val_logits, y_val)
            val_accuracy, val_f1 = classification_metrics(y_val, val_logits)
            row.update(val_loss=val_loss, val_accuracy=val_accuracy, val_f1=val_f1)
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_epoch = epoch
                best_parameters = [parameter.copy() for parameter, _ in model.parameters_and_gradients()]
        else:
            best_epoch = epoch
        history.append(row)
        if validation is not None and epoch - best_epoch >= patience:
            break
    if validation is not None:
        for (parameter, _), best in zip(model.parameters_and_gradients(), best_parameters):
            parameter[...] = best
    return TrainingResult(model, history, best_epoch, best_val_loss)


def candidate_configs(family: str, depth: int, residual: bool) -> list[ModelConfig]:
    activations = ("tanh", "relu") if family == "dense" else ("tanh",)
    gamma_scales = (1.0,) if family == "dense" else GAMMA_SCALES
    return [ModelConfig(family, depth, width, residual, rate, activation, gamma)
            for width, rate, activation, gamma in product(WIDTHS, LEARNING_RATES, activations, gamma_scales)]


def model_name(config: ModelConfig) -> str:
    return f"{config.family}_d{config.depth}_{'residual' if config.residual else 'plain'}"


def select_models(
    x_fit: Matrix, y_fit: Vector, x_val: Matrix, y_val: Vector,
) -> tuple[list[SelectedModel], list[dict]]:
    selected = []
    search_rows = []
    for point, family in ((7, "dense"), (8, "rbf"), (9, "mixed")):
        print(f"\nПункт {point}: подбор гиперпараметров {family.upper()} на валидации (validation)", flush=True)
        for depth, residual in product(DEPTHS, (False, True)):
            best = None
            configs = candidate_configs(family, depth, residual)
            for config in configs:
                started = perf_counter()
                try:
                    model = build_model(config, x_fit, y_fit, SEED)
                    result = fit_model(model, x_fit, y_fit, config.learning_rate, SEED, MAX_EPOCHS, (x_val, y_val))
                    accuracy, f1 = classification_metrics(y_val, result.model.forward(x_val))
                    row = dict(asdict(config), val_loss=result.best_val_loss, val_accuracy=accuracy, val_f1=f1,
                               best_epoch=result.best_epoch, epochs_run=len(result.history), status="ok")
                    if best is None or result.best_val_loss < best.training.best_val_loss:
                        best = SelectedModel(config, result, accuracy, f1)
                except (FloatingPointError, OverflowError) as error:
                    row = dict(asdict(config), status="failed", error=str(error))
                row["seconds"] = perf_counter() - started
                search_rows.append(row)
            if best is None:
                raise RuntimeError(f"All candidates failed for {family}, depth={depth}, residual={residual}.")
            selected.append(best)
            cfg = best.config
            settings = f"width={cfg.width:2d} lr={cfg.learning_rate:g}"
            if cfg.family != "rbf":
                settings += f" activation={cfg.activation}"
            if cfg.family != "dense":
                settings += f" gamma_scale={cfg.gamma_scale:g}"
            print(f"  {model_name(cfg):25s} {settings} "
                  f"epoch={best.training.best_epoch:3d} val_CE={best.training.best_val_loss:.4f} "
                  f"val_F1={best.val_f1:.4f} (вариантов: {len(configs)})", flush=True)
    return selected, search_rows


def evaluate_selected(
    selected: list[SelectedModel], x_train: Matrix, y_train: Vector, x_test: Matrix, y_test: Vector,
) -> tuple[list[dict], list[dict], dict[str, Matrix]]:
    summaries, runs = [], []
    curves = {}
    for choice in selected:
        cfg = choice.config
        name = model_name(cfg)
        losses = []
        model_runs = []
        for seed in REPEAT_SEEDS:
            model = build_model(cfg, x_train, y_train, seed)
            result = fit_model(model, x_train, y_train, cfg.learning_rate, seed, choice.training.best_epoch)
            test_logits = model.forward(x_test)
            accuracy, f1 = classification_metrics(y_test, test_logits)
            test_loss = SoftArgMaxCrossEntropy().forward(test_logits, y_test)
            row = {"model": name, "seed": seed, "test_accuracy": accuracy, "test_f1": f1, "test_loss": test_loss}
            runs.append(row)
            model_runs.append(row)
            losses.append([epoch["train_loss"] for epoch in result.history])
        curves[name] = np.array(losses)
        summary = dict(asdict(cfg), model=name, parameters=model.parameter_count(), best_epoch=choice.training.best_epoch,
                       val_loss=choice.training.best_val_loss, val_accuracy=choice.val_accuracy, val_f1=choice.val_f1)
        for metric in ("test_accuracy", "test_f1", "test_loss"):
            values = [row[metric] for row in model_runs]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        summaries.append(summary)
        print(f"  {name:25s} params={model.parameter_count():5d} epochs={choice.training.best_epoch:3d} "
              f"accuracy={summary['test_accuracy_mean']:.4f} +/- {summary['test_accuracy_std']:.4f} "
              f"macro-F1={summary['test_f1_mean']:.4f} +/- {summary['test_f1_std']:.4f}", flush=True)
    return summaries, runs, curves


def controlled_residual_comparison(
    selected: list[SelectedModel], x_train: Matrix, y_train: Vector, x_test: Matrix, y_test: Vector,
    runs: list[dict],
) -> None:
    print("\nПарное сравнение остаточных связей: одинаковые гиперпараметры, число эпох и seed", flush=True)
    for choice in selected:
        if choice.config.residual:
            continue
        cfg = replace(choice.config, residual=True)
        differences = []
        for seed in REPEAT_SEEDS:
            model = build_model(cfg, x_train, y_train, seed)
            fit_model(model, x_train, y_train, cfg.learning_rate, seed, choice.training.best_epoch)
            _, f1 = classification_metrics(y_test, model.forward(x_test))
            baseline = next(row for row in runs if row["model"] == model_name(choice.config) and row["seed"] == seed)
            differences.append(f1 - baseline["test_f1"])
        print(f"  {cfg.family:5s} depth={cfg.depth}: среднее изменение macro-F1 в парах={np.mean(differences):+.4f}", flush=True)


def save_csv(path: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_experiments(selected: list[SelectedModel], summaries: list[dict], curves: dict[str, Matrix]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(DEPTHS, ("#2864b4", "#d27716", "#23865b")))
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    for row, family in enumerate(("dense", "rbf", "mixed")):
        for choice in selected:
            cfg = choice.config
            if cfg.family != family:
                continue
            label = f"глубина {cfg.depth}, {'с residual' if cfg.residual else 'без residual'}"
            style = "--" if cfg.residual else "-"
            history = choice.training.history
            epochs = [item["epoch"] for item in history]
            for column, metric in enumerate(("train_loss", "val_loss")):
                axes[row, column].plot(epochs, [item[metric] for item in history],
                                       color=colors[cfg.depth], linestyle=style, label=label)
            axes[row, 1].scatter([choice.training.best_epoch], [choice.training.best_val_loss],
                                 color=colors[cfg.depth], s=20)
            losses = curves[model_name(cfg)]
            mean, std = losses.mean(axis=0), losses.std(axis=0)
            axes[row, 2].plot(np.arange(1, len(mean) + 1), mean, color=colors[cfg.depth], linestyle=style, label=label)
            axes[row, 2].fill_between(np.arange(1, len(mean) + 1), mean - std, mean + std,
                                      color=colors[cfg.depth], alpha=0.08)
        for column, title in enumerate(("Подбор: CE на обучении", "Подбор: CE на валидации", "Итоговое обучение: CE (3 seed)")):
            ax = axes[row, column]
            ax.set(title=f"{family.upper()} | {title}", xlabel="Эпоха (epoch)", ylabel="Перекрёстная энтропия (CE)")
            if family != "rbf":
                ax.set_yscale("log")
                ax.set_ylabel("CE (логарифмическая шкала)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Wine: кривые обучения | число эпох выбирается по валидации, test не участвует", fontsize=15)
    fig.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    palette = {"dense": "#2864b4", "rbf": "#d27716", "mixed": "#23865b"}
    lower_limit = max(0.0, min(
        item[f"{metric}_mean"] - item[f"{metric}_std"]
        for item in summaries for metric in ("test_accuracy", "test_f1")
    ) - 0.03)
    for family, residual in product(palette, (False, True)):
        rows = [item for item in summaries if item["family"] == family and item["residual"] == residual]
        label = f"{family.upper()} / {'с residual' if residual else 'без residual'}"
        for column, x_key in enumerate(("depth", "parameters")):
            ordered = sorted(rows, key=lambda item: item[x_key])
            for row, metric in enumerate(("test_accuracy", "test_f1")):
                axes[row, column].errorbar(
                    [item[x_key] for item in ordered], [item[f"{metric}_mean"] for item in ordered],
                    yerr=[item[f"{metric}_std"] for item in ordered], color=palette[family],
                    linestyle="--" if residual else "-", marker="s" if residual else "o", capsize=3, label=label,
                )
                axes[row, column].set(xlabel="Число преобразований (depth)" if column == 0 else "Число обучаемых параметров",
                                      ylabel="Accuracy на тесте" if row == 0 else "Macro-F1 на тесте", ylim=(lower_limit, 1.02))
                if column == 0:
                    axes[row, column].set_xticks(DEPTHS)
                axes[row, column].grid(alpha=0.2)
                axes[row, column].legend(fontsize=8)
    fig.suptitle("Wine: метрики на тесте, среднее ± SD по 3 seed | ось Y увеличена, test одинаковый", fontsize=13)
    fig.savefig(os.path.join(OUTPUT_DIR, "quality_vs_complexity.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    started = perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = load_wine_dataset(DATASET_PATH)
    split = split_dataset(dataset, test_size=TEST_SIZE, seed=SEED)
    preprocessor = fit_preprocessor(split.x_train)
    x_train = preprocessor.transform(split.x_train)
    x_test = preprocessor.transform(split.x_test)

    print("Датасет: Wine")
    print(f"Объектов: {len(dataset.y)}, признаков: {len(dataset.feature_names)}")
    print(f"Классов: {len(dataset.label_names)}")
    print(f"Размеры выборок (объекты, признаки): train={x_train.shape}, test= {x_test.shape}")
    fit_idx, val_idx = stratified_split_indices(split.y_train, VALIDATION_SIZE, SEED + 1)
    tuning_preprocessor = fit_preprocessor(split.x_train[fit_idx])
    x_fit = tuning_preprocessor.transform(split.x_train[fit_idx])
    x_val = tuning_preprocessor.transform(split.x_train[val_idx])
    y_fit, y_val = split.y_train[fit_idx], split.y_train[val_idx]
    print(f"\nВыборки для подбора: обучение (fit)={len(y_fit)}, валидация (validation)={len(y_val)}; отложенный test={len(split.y_test)}")
    print("Функция потерь: численно устойчивая SoftArgMaxCrossEntropy; оптимизатор: Adam; метрики: accuracy и macro-F1.")
    print(f"Глубины (depth)={DEPTHS}; ширины (width)={WIDTHS}; learning_rate={LEARNING_RATES}; максимум эпох={MAX_EPOCHS}", flush=True)
    selected, search_rows = select_models(x_fit, y_fit, x_val, y_val)
    save_csv(os.path.join(OUTPUT_DIR, "search_results.csv"), search_rows)

    print("\nИтоговое обучение и оценка на тесте (все конфигурации уже зафиксированы)", flush=True)
    summaries, runs, curves = evaluate_selected(selected, x_train, split.y_train, x_test, split.y_test)
    save_csv(os.path.join(OUTPUT_DIR, "experiment_summary.csv"), summaries)
    controlled_residual_comparison(selected, x_train, split.y_train, x_test, split.y_test, runs)
    plot_experiments(selected, summaries, curves)
    print(f"Результаты и графики: {OUTPUT_DIR}")
    for filename in ("learning_curves.png", "quality_vs_complexity.png", "experiment_summary.csv"):
        print(f"  {os.path.join(OUTPUT_DIR, filename)}")
    print(f"Общее время: {perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
