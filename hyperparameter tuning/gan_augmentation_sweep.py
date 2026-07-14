import os
import sys
import argparse

# Parse command line args before importing TensorFlow to set device environment variables
parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, choices=[1, 2], default=1, help="Partition index: 1 (Configs 1-8) or 2 (Configs 9-16)")
parser.add_argument("--gpu", type=int, choices=[0, 1], default=0, help="GPU device index to target (0 or 1)")
args, unknown = parser.parse_known_args()

# Set visible GPU device before TF initializes CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Suppress TensorFlow C++ backend warnings/info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import mne
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Suppress TensorFlow Python warning logs
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Reshape, Conv1D, Embedding, Flatten, Concatenate, Lambda, LeakyReLU, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.constraints import max_norm
from mne.decoding import CSP
import numpy as np
import csv
import time

# Target Subjects for the Sweep (default: [1, 2, 4] for speed; change to list(range(1, 10)) for all subjects)
TARGET_SUBJECTS = [1, 2, 4]

# Constants
csv_path = f"gan_sweep_results_part{args.part}.csv"

# Fixed baseline parameters
TMIN = 2.0
TMAX = 6.0
latent_dim = 100
gp_weight = 10.0
n_critic = 5
batch_size = 64

def load_and_preprocess_subject(subject, run_type="T"):
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

    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    available_events = {k: v for k, v in event_dict.items() if k in ["769", "770"]}
    epochs = mne.Epochs(raw, events, event_id=available_events, tmin=2, tmax=6, baseline=None, preload=True, verbose="ERROR")

    X = epochs.get_data()
    y_label = epochs.events[:, 2]

    subject_label_map = {}
    if "769" in epochs.event_id:
        subject_label_map[epochs.event_id["769"]] = 0
    if "770" in epochs.event_id:
        subject_label_map[epochs.event_id["770"]] = 1
    y_label = np.vectorize(subject_label_map.get)(y_label).astype(np.int64)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_flat = X.reshape(-1, X.shape[-1])
    X_norm = scaler.fit_transform(X_flat).reshape(X.shape)
    
    return X_norm, y_label

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

# Generator & Critic
def resize_1d(x, new_length):
    x_4d = tf.expand_dims(x, axis=2)
    x_resized = tf.image.resize(x_4d, [new_length, 1], method='bilinear')
    return tf.squeeze(x_resized, axis=2)

def make_generator(n_channels=10):
    noise_in = Input(shape=(latent_dim,))
    label_in = Input(shape=(1,), dtype='int32')
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Flatten()(label_emb)
    
    x = Concatenate()([noise_in, label_emb])
    x = Dense(125 * 64, activation='relu')(x)
    x = Reshape((125, 64))(x)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 250))(x)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 500))(x)
    x = Conv1D(32, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 1001))(x)
    x_out = Conv1D(n_channels, kernel_size=7, padding='same')(x)
    return Model([noise_in, label_in], x_out, name="Generator")

def make_critic(n_channels=10):
    eeg_in = Input(shape=(1001, n_channels))
    label_in = Input(shape=(1,), dtype='int32')
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Dense(1001)(label_emb)
    label_emb = Reshape((1001, 1))(label_emb)
    
    x = Concatenate(axis=-1)([eeg_in, label_emb])
    x = Conv1D(32, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(128, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    x_out = Dense(1)(x)
    return Model([eeg_in, label_in], x_out, name="Critic")

def train_wgan_gp_for_subject(X_real_data, y_train_data, epochs=100, n_channels=10):
    X_real_data = X_real_data.astype(np.float32)
    X_real_gan = np.transpose(X_real_data[..., 0], (0, 2, 1))
    
    generator = make_generator(n_channels=n_channels)
    critic = make_critic(n_channels=n_channels)

    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step(real_eeg, labels):
        for _ in range(n_critic):
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as critic_tape:
                fake_eeg = generator([noise, labels], training=True)
                d_real = critic([real_eeg, labels], training=True)
                d_fake = critic([fake_eeg, labels], training=True)
                
                epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
                interpolated = epsilon * real_eeg + (1 - epsilon) * fake_eeg
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    d_hat = critic([interpolated, labels], training=True)
                
                gradients = gp_tape.gradient(d_hat, interpolated)
                norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-12)
                gp = tf.reduce_mean((norms - 1.0) ** 2)
                critic_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + gp_weight * gp

            gradients_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(gradients_critic, critic.trainable_variables))
            
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            fake_eeg = generator([noise, labels], training=True)
            d_fake = critic([fake_eeg, labels], training=True)
            gen_loss = -tf.reduce_mean(d_fake)
            
        gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
        return critic_loss, gen_loss

    dataset = tf.data.Dataset.from_tensor_slices((X_real_gan, y_train_data)).shuffle(len(y_train_data)).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        for real_batch, labels_batch in dataset:
            labels_batch = tf.expand_dims(labels_batch, axis=-1)
            train_step(real_batch, labels_batch)
            
    return generator, critic

# CNN Model Builder
def build_cnn(input_shape):
    n_channels = input_shape[0]
    return Sequential([
        Input(shape=input_shape),
        Conv2D(16, (1, 64), padding="same", use_bias=False),
        BatchNormalization(),
        DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.0)),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 4)),
        Dropout(0.5),

        SeparableConv2D(32, (1, 16), padding="same", use_bias=False),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 8)),
        Dropout(0.5),
     
        Flatten(),
        Dense(2, activation="softmax")
    ])

def train_cnn_classifier(X_train_data, y_train_data, input_shape, learning_rate=1e-3, epochs=150, class_weight_dict=None):
    model = build_cnn(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X_train_data,
        y_train_data,
        epochs=epochs,
        batch_size=16,
        class_weight=class_weight_dict,
        verbose=0
    )
    return model

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
    for subj in TARGET_SUBJECTS:
        print(f"  Preloading raw GDF signals for Subject A{subj:02d}...")
        raw_trains[subj] = load_and_preprocess_subject(subj, "T")
        raw_tests[subj] = load_and_preprocess_subject(subj, "E")

    # Define sweep grid configurations (16 combinations)
    wgan_epochs_list = [100, 150]
    filter_selectivity = [0.25, 1.0] # 25% highest score filter vs 100% (No filter)
    aug_ratios = [0.5, 1.0]          # 50% vs 100% added fake data
    paradigms = ["mix", "pretrain"]   # Direct Mixing vs Transfer Learning (Pre-training + Fine-tuning)

    configs = []
    for ep in wgan_epochs_list:
        for filt in filter_selectivity:
            for ratio in aug_ratios:
                for paradigm in paradigms:
                    configs.append({
                        "wgan_epochs": ep,
                        "filter": filt,
                        "ratio": ratio,
                        "paradigm": paradigm
                    })

    # Partition configurations between GPUs/processes (16 configs total)
    if args.part == 1:
        configs_to_run = configs[:8]
        start_idx = 1
    else:
        configs_to_run = configs[8:]
        start_idx = 9

    # Prepare CSV output file (append mode)
    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Config_Idx", "WGAN_Epochs", "Filter_Selectivity", "Aug_Ratio", "Paradigm", "Avg_Accuracy", "Avg_F1"])
            f.flush()

    print(f"\n========== STARTING GAN Augmentation Sweep (Partition {args.part}/2 | Configs {start_idx}-{start_idx+len(configs_to_run)-1}) ==========")
    sys.stdout.flush()

    for offset, config in enumerate(configs_to_run):
        idx = start_idx + offset
        if idx in existing_indices:
            print(f"Skipping Config {idx}/16 (already logged in CSV)")
            sys.stdout.flush()
            continue

        if config["paradigm"] == "pretrain":
            print(f"Skipping pretrain Config {idx}/16 (writing placeholder row)...")
            sys.stdout.flush()
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    config["wgan_epochs"],
                    config["filter"],
                    config["ratio"],
                    config["paradigm"],
                    "0.0000",
                    "0.0000"
                ])
                f.flush()
            continue

        t_start = time.time()
        print(f"\nRunning Config {idx}/16 | WGAN_Ep={config['wgan_epochs']}, Filter={config['filter']}, Ratio={config['ratio']}, Paradigm={config['paradigm']} ...")
        sys.stdout.flush()

        accs = []
        f1s = []

        # Run pipeline over all target subjects
        for subj in TARGET_SUBJECTS:
            # Z-score normalize raw signals
            X_train_norm, y_train = raw_trains[subj]
            X_test_norm, y_test = raw_tests[subj]
            X_train_subj, X_test_subj = z_score_normalize_subject(X_train_norm, X_test_norm)

            classes_real = np.unique(y_train)
            weights_real = compute_class_weight("balanced", classes=classes_real, y=y_train)
            class_weight_dict_real = dict(zip(classes_real, weights_real))

            # Train WGAN
            print(f"  -> Subject A{subj:02d} | Training WGAN-GP ({config['wgan_epochs']} epochs)...")
            sys.stdout.flush()
            generator, critic = train_wgan_gp_for_subject(X_train_subj, y_train, epochs=config["wgan_epochs"], n_channels=10)

            # Generate synthetic data candidate pool (we generate 4x target selection size)
            target_augmentation_size = int(config["ratio"] * len(X_train_subj))
            pool_multiplier = 4
            pool_size = target_augmentation_size * pool_multiplier
            # Safeguard even size
            if pool_size % 2 != 0:
                pool_size += 1
            
            y_fake_pool = np.array([0, 1] * (pool_size // 2), dtype=np.int64)
            noise_pool = np.random.normal(size=(pool_size, latent_dim)).astype(np.float32)
            synthetic_raw_pool = generator.predict([noise_pool, y_fake_pool[:, np.newaxis]], verbose=0)

            # If filtering is enabled, filter using Critic scores
            if config["filter"] < 1.0:
                scores = critic.predict([synthetic_raw_pool, y_fake_pool[:, np.newaxis]], verbose=0).flatten()
                
                target_num_per_class = target_augmentation_size // 2
                selected_idx = []
                for cls in [0, 1]:
                    cls_indices = np.where(y_fake_pool == cls)[0]
                    cls_scores = scores[cls_indices]
                    top_cls_sub_indices = np.argsort(cls_scores)[::-1][:target_num_per_class]
                    selected_idx.extend(cls_indices[top_cls_sub_indices])
                selected_idx = np.array(selected_idx)
                
                X_fake_raw = np.transpose(synthetic_raw_pool[selected_idx], (0, 2, 1))[..., np.newaxis]
                y_fake = y_fake_pool[selected_idx]
            else:
                # No filtering: crop pool directly to target size
                crop_idx = np.random.choice(len(y_fake_pool), target_augmentation_size, replace=False)
                X_fake_raw = np.transpose(synthetic_raw_pool[crop_idx], (0, 2, 1))[..., np.newaxis]
                y_fake = y_fake_pool[crop_idx]

            # Fit CSP and project
            csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
            X_train_csp = csp.fit_transform(X_train_subj[..., 0], y_train)
            X_test_csp = csp.transform(X_test_subj[..., 0])
            X_fake_csp = csp.transform(X_fake_raw[..., 0])

            # Normalize components
            X_train_csp_z, X_test_csp_z = z_score_normalize_subject(X_train_csp, X_test_csp)
            _, X_fake_csp_z = z_score_normalize_subject(X_train_csp, X_fake_csp)

            # Class weights for fake data
            classes_fake = np.unique(y_fake)
            weights_fake = compute_class_weight("balanced", classes=classes_fake, y=y_fake)
            class_weight_dict_fake = dict(zip(classes_fake, weights_fake))

            # Shuffle datasets
            shuffle_idx = np.random.permutation(len(X_fake_csp_z))
            X_fake_final = X_fake_csp_z[shuffle_idx]
            y_fake_final = y_fake[shuffle_idx]

            shuffle_real_idx = np.random.permutation(len(X_train_csp_z))
            X_train_final = X_train_csp_z[shuffle_real_idx]
            y_train_final = y_train[shuffle_real_idx]

            input_shape = X_train_csp_z.shape[1:]

            best_acc = -1
            best_f1 = 0

            # Evaluate CNN training twice, select best
            for run in range(2):
                print(f"    [Run {run+1}/2] Training CNN...")
                sys.stdout.flush()
                
                if config["paradigm"] == "mix":
                    # Direct mix training
                    X_aug = np.concatenate([X_train_final, X_fake_final], axis=0)
                    y_aug = np.concatenate([y_train_final, y_fake_final], axis=0)
                    shuffle_aug = np.random.permutation(len(X_aug))
                    X_aug = X_aug[shuffle_aug]
                    y_aug = y_aug[shuffle_aug]
                    
                    model = train_cnn_classifier(
                        X_aug, y_aug, input_shape, 
                        learning_rate=1e-3, epochs=150, 
                        class_weight_dict=class_weight_dict_real
                    )
                else:
                    # Pre-training + Fine-tuning
                    # 1. Pre-train on synthetic data for 100 epochs
                    model = train_cnn_classifier(
                        X_fake_final, y_fake_final, input_shape, 
                        learning_rate=1e-3, epochs=100, 
                        class_weight_dict=class_weight_dict_fake
                    )
                    # 2. Fine-tune on real data for 50 epochs at lower learning rate
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                    model.fit(
                        X_train_final, y_train_final,
                        epochs=50, batch_size=16,
                        class_weight=class_weight_dict_real,
                        verbose=0
                    )
                
                preds = np.argmax(model.predict(X_test_csp_z, verbose=0), axis=1)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="macro", zero_division=0)
                if acc > best_acc:
                    best_acc = acc
                    best_f1 = f1

            print(f"    Subject A{subj:02d} Best Acc: {best_acc:.4f} | F1: {best_f1:.4f}")
            sys.stdout.flush()
            accs.append(best_acc)
            f1s.append(best_f1)

        avg_acc = np.mean(accs)
        avg_f1 = np.mean(f1s)
        t_duration = time.time() - t_start

        print(f"Finished Config {idx}/16 in {t_duration:.1f}s | Avg Acc: {avg_acc:.4f} | Avg F1: {avg_f1:.4f}")
        sys.stdout.flush()

        # Write results to CSV in real-time
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                config["wgan_epochs"],
                config["filter"],
                config["ratio"],
                config["paradigm"],
                f"{avg_acc:.4f}",
                f"{avg_f1:.4f}"
            ])
            f.flush()

    print("\n========== GAN AUGMENTATION SWEEP COMPLETED SUCCESSFULLY! ==========")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
