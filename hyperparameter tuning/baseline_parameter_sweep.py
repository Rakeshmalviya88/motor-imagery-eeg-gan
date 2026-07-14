import mne
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from mne.decoding import CSP
import numpy as np
import os
import sys
import csv
import time

# Force CPU execution to prevent driver issues
tf.config.set_visible_devices([], 'GPU')

# Constants
csv_path = "baseline_sweep_results.csv"

# Preprocess & Filter loader (Caching base GDF)
def load_raw_and_filter(subject, run_type="T"):
    file_path = f"BCICIV_2a_gdf/A{subject:02d}{run_type}.gdf"
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")

    if run_type == "E":
        mat_path = file_path.replace(".gdf", ".mat")
        mat_data = scipy.io.loadmat(mat_path)
        classlabel = mat_data['classlabel'].flatten()
        label_map = {1: '769', 2: '770', 3: '771', 4: '772'}
        test_annotations = [label_map[lbl] for lbl in classlabel]
        descriptions = raw.annotations.description.copy()
        idx = 0
        for i, desc in enumerate(descriptions):
            if desc == '783':
                descriptions[i] = test_annotations[idx]
                idx += 1
        raw.annotations.description = descriptions

    eog_channels = ["EOG-left", "EOG-central", "EOG-right"]
    present_eog = {ch: "eog" for ch in eog_channels if ch in raw.ch_names}
    if present_eog:
        raw.set_channel_types(present_eog)

    gdf_to_standard = {
        'EEG-Fz': 'EEG-Fz', 'EEG-0': 'EEG-FC3', 'EEG-1': 'EEG-FC1', 'EEG-2': 'EEG-FCz', 'EEG-3': 'EEG-FC2',
        'EEG-4': 'EEG-FC4', 'EEG-5': 'EEG-C5', 'EEG-C3': 'EEG-C3', 'EEG-6': 'EEG-C1', 'EEG-Cz': 'EEG-Cz',
        'EEG-7': 'EEG-C2', 'EEG-C4': 'EEG-C4', 'EEG-8': 'EEG-C6', 'EEG-9': 'EEG-CP3', 'EEG-10': 'EEG-CP1',
        'EEG-11': 'EEG-CPz', 'EEG-12': 'EEG-CP2', 'EEG-13': 'EEG-CP4', 'EEG-14': 'EEG-P1', 'EEG-Pz': 'EEG-Pz',
        'EEG-15': 'EEG-P2', 'EEG-16': 'EEG-POz'
    }
    rename_dict = {k: v for k, v in gdf_to_standard.items() if k in raw.ch_names}
    raw.rename_channels(rename_dict)

    target_channels = ['EEG-FC3', 'EEG-FC1', 'EEG-C3', 'EEG-C1', 'EEG-C4', 'EEG-CP1', 'EEG-CPz', 'EEG-P1', 'EEG-P2', 'EEG-POz']
    raw.pick(target_channels)

    raw.notch_filter(50, verbose="ERROR")
    raw.filter(8, 30, verbose="ERROR")
    return raw

# Dynamic trial slicing and MinMaxScaler scaling
def epoch_and_scale(raw, tmin, tmax):
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    available_events = {
        k: v for k, v in event_dict.items()
        if k in ["769", "770"]
    }
    epochs = mne.Epochs(
        raw,
        events,
        event_id=available_events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR"
    )
    X = epochs.get_data()
    y = epochs.events[:, 2]

    subject_label_map = {}
    if "769" in epochs.event_id:
        subject_label_map[epochs.event_id["769"]] = 0
    if "770" in epochs.event_id:
        subject_label_map[epochs.event_id["770"]] = 1
    y = np.vectorize(subject_label_map.get)(y).astype(np.int64)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_flat = X.reshape(-1, X.shape[-1])
    X_norm = scaler.fit_transform(X_flat).reshape(X.shape)
    
    return X_norm, y

# Trial Z-scoring
def z_score_normalize_subject(X_train_data, X_test_data):
    X_train_data = X_train_data.astype(np.float32)
    mean_train = X_train_data.mean(axis=(1, 2), keepdims=True)
    std_train = X_train_data.std(axis=(1, 2), keepdims=True)
    X_train_data = (X_train_data - mean_train) / (std_train + 1e-8)
    X_train_data = np.nan_to_num(X_train_data)

    X_test_data = X_test_data.astype(np.float32)
    mean_test = X_test_data.mean(axis=(1, 2), keepdims=True)
    std_test = X_test_data.std(axis=(1, 2), keepdims=True)
    X_test_data = (X_test_data - mean_test) / (std_test + 1e-8)
    X_test_data = np.nan_to_num(X_test_data)

    X_train_data = X_train_data[..., np.newaxis]
    X_test_data = X_test_data[..., np.newaxis]
    return X_train_data, X_test_data

# CNN Model Builder
def build_cnn(input_shape, kernel_length=64, dropout_rate=0.5):
    n_channels = input_shape[0]
    return Sequential([
        Input(shape=input_shape),
        Conv2D(16, (1, kernel_length), padding="same", use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.0)),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 4)),
        Dropout(dropout_rate),
        SeparableConv2D(32, (1, 16), padding="same", use_bias=False),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 8)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(2, activation="softmax")
    ])

def train_cnn_classifier(X_train_data, y_train_data, kernel_length=64, dropout_rate=0.5, epochs=150, verbose=0):
    class_labels = np.unique(y_train_data)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=y_train_data
    )
    class_weight_dict = dict(zip(class_labels, class_weights))
    
    model = build_cnn(X_train_data.shape[1:], kernel_length, dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X_train_data,
        y_train_data,
        epochs=epochs,
        batch_size=16,
        class_weight=class_weight_dict,
        verbose=verbose
    )
    return model

# Main Sweep Execution
def main():
    # Read existing config indices from CSV to enable resume
    existing_indices = set()
    file_exists = os.path.exists(csv_path)
    if file_exists:
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        existing_indices.add(int(row[0]))
        except Exception as e:
            print(f"Error reading CSV header / indices: {e}")

    print("========== PRELOADING EEG SIGNAL DATABASE ==========")
    raw_trains = {}
    raw_tests = {}
    for subj in range(1, 10):
        print(f"  Preloading raw GDF signals for Subject A{subj:02d}...")
        raw_trains[subj] = load_raw_and_filter(subj, "T")
        raw_tests[subj] = load_raw_and_filter(subj, "E")

    # Define sweep grid configurations (16 combinations)
    csp_regs = [None, 'ledoit_wolf']
    time_windows = [(2.0, 6.0), (2.5, 5.5)]
    kernel_lengths = [64, 128]
    dropout_rates = [0.4, 0.5]

    configs = []
    for reg in csp_regs:
        for window in time_windows:
            for kernel in kernel_lengths:
                for dropout in dropout_rates:
                    configs.append({
                        "csp_reg": reg,
                        "tmin": window[0],
                        "tmax": window[1],
                        "kernel_length": kernel,
                        "dropout_rate": dropout
                    })

    # Prepare CSV output file (append mode)
    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Config_Idx", "CSP_Reg", "Time_Window", "Kernel_Length", "Dropout_Rate", "Avg_Accuracy", "Avg_F1"])
            f.flush()

    print(f"\n========== STARTING Baseline Parameter Sweep (1 CNN Run Per Subject) ==========")
    sys.stdout.flush()

    for idx, config in enumerate(configs, 1):
        if idx in existing_indices:
            print(f"Skipping Config {idx}/16 (already logged in CSV)")
            sys.stdout.flush()
            continue

        t_start = time.time()
        print(f"\nRunning Config {idx}/16 | reg={config['csp_reg']}, window=({config['tmin']}, {config['tmax']}), kernel={config['kernel_length']}, dropout={config['dropout_rate']} ...")
        sys.stdout.flush()

        accs = []
        f1s = []

        # Run pipeline over all 9 subjects
        for subj in range(1, 10):
            # 1. Epoch and Scale dynamically on cached raw structures
            X_train_norm, y_train = epoch_and_scale(raw_trains[subj], config["tmin"], config["tmax"])
            X_test_norm, y_test = epoch_and_scale(raw_tests[subj], config["tmin"], config["tmax"])

            # 2. Trial Z-scoring
            X_train_subj, X_test_subj = z_score_normalize_subject(X_train_norm, X_test_norm)

            # 3. Fit CSP on z-scored real training data
            csp = CSP(n_components=4, reg=config["csp_reg"], log=None, transform_into='csp_space')
            X_train_csp = csp.fit_transform(X_train_subj[..., 0], y_train)
            X_test_csp = csp.transform(X_test_subj[..., 0])

            # 4. Normalise components
            X_train_csp_z, X_test_csp_z = z_score_normalize_subject(X_train_csp, X_test_csp)

            # Shuffle training set
            shuffle_idx = np.random.permutation(len(X_train_csp_z))
            X_train_final = X_train_csp_z[shuffle_idx]
            y_train_final = y_train[shuffle_idx]

            # 5. Train CNN once (single run) for maximum speed
            model = train_cnn_classifier(X_train_final, y_train_final, config["kernel_length"], config["dropout_rate"], epochs=150, verbose=0)
            preds = np.argmax(model.predict(X_test_csp_z, verbose=0), axis=1)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)

            accs.append(acc)
            f1s.append(f1)

        avg_acc = np.mean(accs)
        avg_f1 = np.mean(f1s)
        t_duration = time.time() - t_start

        print(f"Finished Config {idx}/16 in {t_duration:.1f}s | Avg Acc: {avg_acc:.4f} | Avg F1: {avg_f1:.4f}")
        sys.stdout.flush()

        # Write out to CSV immediately (Real-Time output)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                str(config["csp_reg"]),
                f"({config['tmin']}, {config['tmax']})",
                config["kernel_length"],
                config["dropout_rate"],
                f"{avg_acc:.4f}",
                f"{avg_f1:.4f}"
            ])
            f.flush()

    print("\n========== BASELINE PARAMETER SWEEP COMPLETED SUCCESSFULLY! ==========")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
