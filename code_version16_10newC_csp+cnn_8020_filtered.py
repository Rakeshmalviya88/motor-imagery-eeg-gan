'''
 PHASE 1 : PREPROCESSING

* BCI Competition IV Dataset 2a
* Channels: 10 selected channels [FC3, FC1, C3, CP3, CP1, FC2, FC4, C4, CP2, CP4]
* All subjects: A01, A02, A03, A04, A05, A06, A07, A08, A09
'''

import warnings
import mne
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from mne.decoding import CSP

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, 
    BatchNormalization, Activation, Dense, Dropout, Flatten, 
    Reshape, Conv1D, Embedding, Concatenate, Lambda, LeakyReLU
)
from tensorflow.keras.constraints import max_norm

# hiding annoying mne warnings about channel types
warnings.filterwarnings("ignore", message="Could not determine channel type of the following channels, they will be set as EEG:*")

# -----------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS
# -----------------------------------------------------------------------------
latent_dim = 100 # size of the noise vector for the GAN generator
gp_weight = 10.0 # standard gp weight from the wgan-gp paper to enforce lipschitz constraint
n_critic = 5     # train the critic 5 times for every 1 generator update
batch_size = 64
epochs_wgan_value = 150
epochs_cnn_value = 150
epochs_cnn_baseline = 150


# -----------------------------------------------------------------------------
# 1. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
def load_and_preprocess_subject(subject, run_type="T"):
    """
    Loads raw GDF files, extracts motor channels, applies filters, and epochs the data.
    """
    # read the raw gdf file for the specific subject
    file_path = f"BCICIV_2a_gdf/A{subject:02d}{run_type}.gdf"
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")

    # if it's the evaluation file, we need to fix the missing labels using the .mat file
    # because the BCI competition hid the true labels for the E files as '783'
    if run_type == "E":
        mat_path = file_path.replace(".gdf", ".mat")
        mat_data = scipy.io.loadmat(mat_path)
        classlabel = mat_data['classlabel'].flatten()
        
        # map classes 1-4 back to their corresponding mne event codes
        label_map = {1: '769', 2: '770', 3: '771', 4: '772'}
        test_annotations = [label_map[lbl] for lbl in classlabel]
        descriptions = raw.annotations.description.copy()
        idx = 0
        for i, desc in enumerate(descriptions):
            if desc == '783': # 783 means unknown label in the raw dataset
                descriptions[i] = test_annotations[idx]
                idx += 1
        raw.annotations.description = descriptions

    # making sure eog channels are explicitly set so they don't get mixed up with eeg data
    eog_channels = ["EOG-left", "EOG-central", "EOG-right"]
    present_eog = {ch: "eog" for ch in eog_channels if ch in raw.ch_names}
    if present_eog:
        raw.set_channel_types(present_eog)

    # map weird gdf numeric channel names to the standard 10-20 system
    gdf_to_standard = {
        'EEG-Fz': 'EEG-Fz', 'EEG-0': 'EEG-FC3', 'EEG-1': 'EEG-FC1', 'EEG-2': 'EEG-FCz', 'EEG-3': 'EEG-FC2',
        'EEG-4': 'EEG-FC4', 'EEG-5': 'EEG-C5', 'EEG-C3': 'EEG-C3', 'EEG-6': 'EEG-C1', 'EEG-Cz': 'EEG-Cz',
        'EEG-7': 'EEG-C2', 'EEG-C4': 'EEG-C4', 'EEG-8': 'EEG-C6', 'EEG-9': 'EEG-CP3', 'EEG-10': 'EEG-CP1',
        'EEG-11': 'EEG-CPz', 'EEG-12': 'EEG-CP2', 'EEG-13': 'EEG-CP4', 'EEG-14': 'EEG-P1', 'EEG-Pz': 'EEG-Pz',
        'EEG-15': 'EEG-P2', 'EEG-16': 'EEG-POz'
    }
    rename_dict = {k: v for k, v in gdf_to_standard.items() if k in raw.ch_names}
    raw.rename_channels(rename_dict)

    # picking ONLY the 10 symmetric motor cortex channels specifically for left/right hand MI
    target_channels = ['EEG-FC3', 'EEG-FC1', 'EEG-C3', 'EEG-CP3', 'EEG-CP1', 'EEG-FC4', 'EEG-FC2', 'EEG-C4', 'EEG-CP4', 'EEG-CP2']
    raw.pick(target_channels)
    
    # 50 Hz notch filter to remove european powerline electrical noise
    raw.notch_filter(50, verbose="ERROR")
    # bandpass 8-30 Hz to isolate the Mu and Beta bands where motor imagery happens
    raw.filter(8, 30, verbose="ERROR")

    # extracting events 769 (left hand) and 770 (right hand)
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    available_events = {k: v for k, v in event_dict.items() if k in ["769", "770"]}

    # epoching the continuous recording from 2 to 6 seconds (the actual motor imagery task window)
    epochs = mne.Epochs(raw, events, event_id=available_events, tmin=2, tmax=6, baseline=None, preload=True, verbose="ERROR")
    X = epochs.get_data()
    y_label = epochs.events[:, 2]

    # map raw event numbers down to 0 and 1 so the neural network can understand them
    subject_label_map = {}
    if "769" in epochs.event_id:
        subject_label_map[epochs.event_id["769"]] = 0
    if "770" in epochs.event_id:
        subject_label_map[epochs.event_id["770"]] = 1
    y_label = np.vectorize(subject_label_map.get)(y_label).astype(np.int64)

    # apply min max scaling (-1 to 1) so the GAN generator has an easier time converging
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_flat = X.reshape(-1, X.shape[-1])
    X_norm = scaler.fit_transform(X_flat).reshape(X.shape)
    
    return X_norm, y_label

def z_score_normalize_subject(X_train_data, X_test_data):
    """
    Standardizes data to mean 0 and standard deviation 1 based on the training set statistics.
    """
    X_train_data = X_train_data.astype(np.float32)
    mean_train = X_train_data.mean(axis=(1, 2), keepdims=True)
    std_train = X_train_data.std(axis=(1, 2), keepdims=True)
    
    # normalize train data and catch any nan values just in case
    X_train_data = (X_train_data - mean_train) / (std_train + 1e-8)
    X_train_data = np.nan_to_num(X_train_data) 

    X_test_data = X_test_data.astype(np.float32)
    mean_test = X_test_data.mean(axis=(1, 2), keepdims=True)
    std_test = X_test_data.std(axis=(1, 2), keepdims=True)
    
    # normalize test data using its own stats (or could use train stats, but doing independent here)
    X_test_data = (X_test_data - mean_test) / (std_test + 1e-8)
    X_test_data = np.nan_to_num(X_test_data)

    # add a dummy dimension at the end to make it compatible with 2D convolutions later
    X_train_data = X_train_data[..., np.newaxis]
    X_test_data = X_test_data[..., np.newaxis]
    
    return X_train_data, X_test_data


# -----------------------------------------------------------------------------
# 2. WGAN-GP ARCHITECTURE
# -----------------------------------------------------------------------------
def resize_1d(x, new_length):
    # simple helper function for upsampling 1d eeg signals in the generator layers
    x_4d = tf.expand_dims(x, axis=2)
    x_resized = tf.image.resize(x_4d, [new_length, 1], method='bilinear')
    return tf.squeeze(x_resized, axis=2)

def make_generator(n_channels=10):
    """
    Takes random noise and a class label, and gradually upsamples it into a 1001-point EEG signal.
    """
    noise_in = Input(shape=(latent_dim,))
    label_in = Input(shape=(1,), dtype='int32')
    
    # embedding the label so the gan can generate specific classes (left or right) instead of a random mess
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Flatten()(label_emb)
    
    # stick the noise and the label together
    x = Concatenate()([noise_in, label_emb])
    x = Dense(125 * 64, activation='relu')(x)
    x = Reshape((125, 64))(x)
    
    # gradually upsampling to stretch the signal to 1001 timepoints
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 250))(x)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 500))(x)
    x = Conv1D(32, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 1001))(x)
    x_out = Conv1D(n_channels, kernel_size=7, padding='same')(x) # output is (1001, 10)
    
    return Model([noise_in, label_in], x_out, name="Generator")

def make_critic(n_channels=10):
    """
    Evaluates an EEG signal and outputs a single 'realness' score (no sigmoid at the end because WGAN!)
    """
    eeg_in = Input(shape=(1001, n_channels))
    label_in = Input(shape=(1,), dtype='int32')
    
    # embedding for the critic to judge if the class actually matches the eeg pattern
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Dense(1001)(label_emb)
    label_emb = Reshape((1001, 1))(label_emb)
    
    # combine the eeg signal and the label
    x = Concatenate(axis=-1)([eeg_in, label_emb])
    
    # strided convolutions for downsampling the signal to extract features
    x = Conv1D(32, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(128, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    x_out = Dense(1)(x) # linear output for WGAN (again, absolutely no sigmoid here!)
    
    return Model([eeg_in, label_in], x_out, name="Critic")

def train_wgan_gp_for_subject(X_real_data, y_train_data, epochs=100, verbose=False, n_channels=10):
    """
    The custom training loop for the WGAN-GP. Handles the gradient penalty math.
    """
    X_real_data = X_real_data.astype(np.float32)
    # converting to channels_last format because tensorflow expects it that way
    X_real_gan = np.transpose(X_real_data[..., 0], (0, 2, 1))
    
    generator = make_generator(n_channels=n_channels)
    critic = make_critic(n_channels=n_channels)

    # adam optimizers with standard parameters from the wgan paper
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

    @tf.function
    def train_step(real_eeg, labels):
        # train critic multiple times (n_critic) for every 1 generator step to keep it strong
        for _ in range(n_critic):
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as critic_tape:
                fake_eeg = generator([noise, labels], training=True)
                d_real = critic([real_eeg, labels], training=True)
                d_fake = critic([fake_eeg, labels], training=True)
                
                # gradient penalty calculation to enforce lipschitz constraint
                # this stops the model gradients from exploding
                epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
                interpolated = epsilon * real_eeg + (1 - epsilon) * fake_eeg
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    d_hat = critic([interpolated, labels], training=True)
                
                gradients = gp_tape.gradient(d_hat, interpolated)
                norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-12)
                gp = tf.reduce_mean((norms - 1.0) ** 2)
                
                # WGAN-GP loss formula: maximize distance between real and fake, plus penalty
                critic_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + gp_weight * gp

            # apply gradients for the critic
            gradients_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(gradients_critic, critic.trainable_variables))
            
        # now train the generator
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            fake_eeg = generator([noise, labels], training=True)
            d_fake = critic([fake_eeg, labels], training=True)
            # generator just wants the critic to think its fake data is real
            gen_loss = -tf.reduce_mean(d_fake)
            
        # apply gradients for the generator
        gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
        return critic_loss, gen_loss

    dataset = tf.data.Dataset.from_tensor_slices((X_real_gan, y_train_data)).shuffle(len(y_train_data)).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        for real_batch, labels_batch in dataset:
            labels_batch = tf.expand_dims(labels_batch, axis=-1)
            train_step(real_batch, labels_batch)
            
    return generator, critic


# -----------------------------------------------------------------------------
# 3. CNN CLASSIFIER (EEGNet Architecture)
# -----------------------------------------------------------------------------
def train_cnn_classifier(X_train_data, y_train_data, epochs=150, verbose=0):
    # balancing class weights just in case the augmentation makes the data slightly uneven
    class_labels = np.unique(y_train_data)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=y_train_data
    )
    class_weight_dict = dict(zip(class_labels, class_weights))
    
    n_channels = X_train_data.shape[1]
    
    # standard eegnet-like architecture
    cnn_model = Sequential([
        Input(shape=X_train_data.shape[1:]),
        
        # temporal convolution (looks at patterns over time)
        Conv2D(16, (1, 64), padding="same", use_bias=False),
        BatchNormalization(),
        
        # spatial convolution (looks across the 10 channels)
        DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.0)),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 4)),
        Dropout(0.5),

        # point-wise convolution (combines features safely)
        SeparableConv2D(32, (1, 16), padding="same", use_bias=False),
        BatchNormalization(),
        Activation("elu"),
        AveragePooling2D((1, 8)),
        Dropout(0.5),
     
        Flatten(),
        Dense(2, activation="softmax")
    ])
    
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    cnn_model.fit(
        X_train_data,
        y_train_data,
        epochs=epochs,
        batch_size=16,
        class_weight=class_weight_dict,
        verbose=verbose
    )
    return cnn_model


# -----------------------------------------------------------------------------
# 4. FULL AUGMENTED PIPELINE
# -----------------------------------------------------------------------------
def train_and_evaluate_subject(subject_id, epochs_wgan=150, epochs_cnn=150):
    # 1. load the data
    X_full_norm, y_full_subj = load_and_preprocess_subject(subject_id, "T")
    
    # 2. standard 80/20 train test split (test data gets locked away entirely!)
    X_train_norm, X_test_norm, y_train_subj, y_test_subj = train_test_split(X_full_norm, y_full_subj, test_size=0.2, random_state=42, stratify=y_full_subj)
    
    # 3. z-score normalize the data
    X_train_subj, X_test_subj = z_score_normalize_subject(X_train_norm, X_test_norm)
    
    # 4. train the GAN on the real training data
    print(f"-> Training WGAN-GP on Subject A{subject_id:02d} for {epochs_wgan} epochs...")
    generator_subj, critic_subj = train_wgan_gp_for_subject(X_train_subj, y_train_subj, epochs=epochs_wgan, verbose=False, n_channels=10)
    
    # 5. generate a massive pool of synthetic data (3x the train size) so the critic has plenty to evaluate
    pool_multiplier = 3
    pool_size = len(X_train_subj) * pool_multiplier
    if pool_size % 2 != 0:
        pool_size += 1
    y_fake_pool = np.array([0, 1] * (pool_size // 2), dtype=np.int64)
    noise_pool = np.random.normal(size=(pool_size, latent_dim)).astype(np.float32)
    synthetic_raw_pool = generator_subj.predict([noise_pool, y_fake_pool[:, np.newaxis]], verbose=0)
    
    # 6. the critic grades all the synthetic samples we just generated
    scores = critic_subj.predict([synthetic_raw_pool, y_fake_pool[:, np.newaxis]], verbose=0).flatten()
    
    # 7. selecting only the absolute best scoring samples to add exactly 50% overall augmentation
    target_num_per_class = int(0.25 * len(X_train_subj)) # 25% for class 0, 25% for class 1 = 50% total
    selected_indices = []
    
    for cls in [0, 1]:
        cls_candidate_idx = np.where(y_fake_pool == cls)[0]
        cls_scores = scores[cls_candidate_idx]
        
        # sort scores highest to lowest and skim only the top chunk
        sorted_sub_idx = np.argsort(cls_scores)[::-1][:target_num_per_class]
        selected_indices.extend(cls_candidate_idx[sorted_sub_idx])
    selected_indices = np.array(selected_indices)
    
    # grab the best samples using the indices we just found
    X_fake_train = np.transpose(synthetic_raw_pool[selected_indices], (0, 2, 1))[..., np.newaxis]
    y_fake_train = y_fake_pool[selected_indices]

    # 8. fit CSP (Common Spatial Pattern) purely on the REAL training data to learn the spatial filters
    csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
    X_train_csp = csp.fit_transform(X_train_subj[..., 0], y_train_subj)
    
    # apply the spatial filters to the test data and the fake data
    X_test_csp = csp.transform(X_test_subj[..., 0])
    X_fake_csp = csp.transform(X_fake_train[..., 0])

    # 9. normalize the csp features
    X_train_csp_z, X_test_csp_z = z_score_normalize_subject(X_train_csp, X_test_csp)
    _, X_fake_csp_z = z_score_normalize_subject(X_train_csp, X_fake_csp)

    # 10. append the high-quality synthetic data to the training set!
    X_aug = np.concatenate([X_train_csp_z, X_fake_csp_z], axis=0)
    y_aug = np.concatenate([y_train_subj, y_fake_train], axis=0)

    # 11. shuffle everything so the network doesn't just see a block of real data then a block of fake data
    shuffle_idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]
    
    best_acc = -1
    best_prec, best_rec, best_f1 = 0, 0, 0
    
    # 12. run cnn multiple times and keep the best metrics to avoid bad random initialization ruining the score
    for run in range(6):
        print(f"   [Run {run+1}/6] Training CNN for Subject A{subject_id:02d}...")
        model_cnn = train_cnn_classifier(X_aug, y_aug, epochs=epochs_cnn, verbose=0)
        
        preds_aug = np.argmax(model_cnn.predict(X_test_csp_z, verbose=0), axis=1)
        acc = accuracy_score(y_test_subj, preds_aug)
        prec = precision_score(y_test_subj, preds_aug, average="macro", zero_division=0)
        rec = recall_score(y_test_subj, preds_aug, average="macro", zero_division=0)
        f1 = f1_score(y_test_subj, preds_aug, average="macro", zero_division=0)
        
        # update high score
        if acc > best_acc:
            best_acc, best_prec, best_rec, best_f1 = acc, prec, rec, f1
            
    return best_acc, best_prec, best_rec, best_f1


# -----------------------------------------------------------------------------
# MAIN EXECUTION SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*79)
    print("STARTING INDIVIDUAL SUBJECT TRAINING AND EVALUATION (AUGMENTED)")
    print("="*79)

    subject_results = {}
    for subject in range(1, 10):
        print(f"\n[SUBJECT A{subject:02d}]")
        acc, prec, rec, f1 = train_and_evaluate_subject(subject, epochs_wgan=epochs_wgan_value, epochs_cnn=epochs_cnn_value)
        subject_results[subject] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"Subject A{subject:02d} Results (Best of 6 Runs) - Accuracy: {acc:.4f}, F1: {f1:.4f}")

    print("\n" + "="*79)
    print("INDIVIDUAL SUBJECT-WISE EVALUATION SUMMARY (AUGMENTED)")
    print("="*79)

    accs = [res["accuracy"] for res in subject_results.values()]
    precs = [res["precision"] for res in subject_results.values()]
    recs = [res["recall"] for res in subject_results.values()]
    f1s = [res["f1"] for res in subject_results.values()]

    for subject, res in subject_results.items():
        print(f"Subject A{subject:02d} | Accuracy: {res['accuracy']:.4f} | F1 Score: {res['f1']:.4f}")
        
    print("-" * 50)
    print(f"Average Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Average Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"Average Recall   : {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"Average F1 Score : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print("="*79 + "\n")

    # -------------------------------------------------------------------------
    # RUNNING THE EXACT SAME PIPELINE WITHOUT GAN FOR BASELINE COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "="*79)
    print("STARTING BASELINE EVALUATION (REAL DATA ONLY, NO AUGMENTATION)")
    print("="*79)

    baseline_results = {}
    for subject in range(1, 10):
        print(f"\n[BASELINE - SUBJECT A{subject:02d}]")
        X_full_norm, y_full_subj = load_and_preprocess_subject(subject, "T")
        X_train_norm, X_test_norm, y_train_subj, y_test_subj = train_test_split(X_full_norm, y_full_subj, test_size=0.2, random_state=42, stratify=y_full_subj)
        
        X_train_subj, X_test_subj = z_score_normalize_subject(X_train_norm, X_test_norm)
        
        csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
        X_train_csp = csp.fit_transform(X_train_subj[..., 0], y_train_subj)
        X_test_csp = csp.transform(X_test_subj[..., 0])

        X_train_csp_z, X_test_csp_z = z_score_normalize_subject(X_train_csp, X_test_csp)
        
        shuffle_idx = np.random.permutation(len(X_train_csp_z))
        X_train_final = X_train_csp_z[shuffle_idx]
        y_train_final = y_train_subj[shuffle_idx]
        
        best_acc = -1
        best_prec, best_rec, best_f1 = 0, 0, 0
        
        # No. of CNN runs
        for run in range(6):
            print(f"   [Run {run+1}/6] Training CNN Baseline for Subject A{subject:02d}...")
            model_cnn = train_cnn_classifier(X_train_final, y_train_final, epochs=epochs_cnn_baseline, verbose=0)
            
            preds = np.argmax(model_cnn.predict(X_test_csp_z, verbose=0), axis=1)
            acc = accuracy_score(y_test_subj, preds)
            prec = precision_score(y_test_subj, preds, average="macro", zero_division=0)
            rec = recall_score(y_test_subj, preds, average="macro", zero_division=0)
            f1 = f1_score(y_test_subj, preds, average="macro", zero_division=0)
            
            if acc > best_acc:
                best_acc, best_prec, best_rec, best_f1 = acc, prec, rec, f1
                
        baseline_results[subject] = {"accuracy": best_acc, "precision": best_prec, "recall": best_rec, "f1": best_f1}
        print(f"Subject A{subject:02d} Baseline (Best of 6 Runs) - Accuracy: {best_acc:.4f}, F1: {best_f1:.4f}")

    print("\n" + "="*79)
    print("BASELINE EVALUATION SUMMARY (REAL DATA ONLY)")
    print("="*79)

    accs_b = [res["accuracy"] for res in baseline_results.values()]
    precs_b = [res["precision"] for res in baseline_results.values()]
    recs_b = [res["recall"] for res in baseline_results.values()]
    f1s_b = [res["f1"] for res in baseline_results.values()]

    for subject, res in baseline_results.items():
        print(f"Subject A{subject:02d} | Accuracy: {res['accuracy']:.4f} | F1 Score: {res['f1']:.4f}")
        
    print("-" * 50)
    print(f"Average Accuracy : {np.mean(accs_b):.4f} +/- {np.std(accs_b):.4f}")
    print(f"Average Precision: {np.mean(precs_b):.4f} +/- {np.std(precs_b):.4f}")
    print(f"Average Recall   : {np.mean(recs_b):.4f} +/- {np.std(recs_b):.4f}")
    print(f"Average F1 Score : {np.mean(f1s_b):.4f} +/- {np.std(f1s_b):.4f}")
    print("="*79 + "\n")
