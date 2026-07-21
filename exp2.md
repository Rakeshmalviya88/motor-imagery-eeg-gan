# Explanation of Phase 2: Generative Adversarial Network (WGAN-GP)

This document provides a line-by-line explanation of the Phase 2 code cell in [code_version16_10newC_csp+cnn_8020_filtered.ipynb](file:///d:/My%20Projects/internship/code_version16_10newC_csp+cnn_8020_filtered.ipynb#L176-L308). This cell is responsible for implementing and training a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to generate synthetic EEG signals for motor imagery data augmentation.

---

## Table of Contents
1. [Overview of WGAN-GP](#1-overview-of-wgan-gp)
2. [Helper Function: `resize_1d`](#2-helper-function-resize_1d)
3. [The Generator Model: `make_generator`](#3-the-generator-model-make_generator)
4. [The Critic Model: `make_critic`](#4-the-critic-model-make_critic)
5. [The Training Loop: `train_wgan_gp_for_subject`](#5-the-training-loop-train_wgan_gp_for_subject)
6. [Why WGAN-GP for EEG Augmentation?](#6-why-wgan-gp-for-eeg-augmentation)

---

## 1. Overview of WGAN-GP

In a standard GAN, a **Generator** tries to create realistic data from random noise, while a **Discriminator** tries to classify samples as either real or fake. Standard GANs are notoriously hard to train due to:
* **Mode Collapse:** The generator learns to produce only a limited set of outputs (e.g., only one pattern of EEG).
* **Vanishing Gradients:** If the discriminator gets too good too fast, the generator stops receiving useful gradient signals.

**WGAN-GP (Wasserstein GAN with Gradient Penalty)** solves this by:
1. Replacing the classification probability (0 to 1) with a continuous **"realness" score** output by a **Critic** (instead of a Discriminator).
2. Optimizing the **Wasserstein Distance** (also known as Earth Mover's Distance), which measures how far the generated distribution is from the real distribution.
3. Adding a **Gradient Penalty** to enforce the **1-Lipschitz continuity condition** (meaning the Critic's gradients must have a norm of at most 1). This mathematical constraint ensures stable training and prevents exploding/vanishing gradients.

---

## 2. Helper Function: `resize_1d`

```python
def resize_1d(x, new_length):
    # simple helper function for upsampling 1d eeg signals in the generator layers
    x_4d = tf.expand_dims(x, axis=2)
    x_resized = tf.image.resize(x_4d, [new_length, 1], method='bilinear')
    return tf.squeeze(x_resized, axis=2)
```

### Line-by-Line Breakdown:
* **`def resize_1d(x, new_length):`**
  Defines a function to stretch/upsample a 1D tensor along its sequence length.
* **`x_4d = tf.expand_dims(x, axis=2)`**
  TensorFlow's image resizing utility `tf.image.resize` requires a 4D tensor representing `[batch_size, height, width, channels]`. EEG signals are 3D: `[batch_size, sequence_length, channels]`. This adds a dummy dimension at index 2 (representing a width of 1) to convert the shape to `[batch_size, sequence_length, 1, channels]`.
* **`x_resized = tf.image.resize(x_4d, [new_length, 1], method='bilinear')`**
  Resizes the 4D tensor to have a new height (`new_length`) and a width of 1 using bilinear interpolation. This smoothly interpolates new values to stretch the sequence.
* **`return tf.squeeze(x_resized, axis=2)`**
  Removes the dummy width dimension (index 2) to return the tensor to its original 3D shape `[batch_size, new_length, channels]`.

### Why do this?
Standard upsampling layers in deep learning (like `UpSampling1D`) perform simple replication or nearest-neighbor interpolation, which can introduce blocky artifacts in continuous signals like EEG. Leveraging bilinear interpolation via image resizing creates smooth, realistic wave transitions.

---

## 3. The Generator Model: `make_generator`

The Generator takes random noise and a target class label (Left or Right Hand) and gradually projects/upsamples it into a full $1001 \times 10$ EEG trial (1001 timepoints across 10 channels).

```python
def make_generator(n_channels=10):
    noise_in = Input(shape=(latent_dim,))
    label_in = Input(shape=(1,), dtype='int32')
```
* **`noise_in = Input(shape=(latent_dim,))`**
  Defines the input layer for the random noise vector (a 1D vector of size `latent_dim = 100` drawn from a standard normal distribution).
* **`label_in = Input(shape=(1,), dtype='int32')`**
  Defines the input layer for the conditional class label (0 for Left Hand Motor Imagery, 1 for Right Hand Motor Imagery).

```python
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Flatten()(label_emb)
```
* **`Embedding(input_dim=2, output_dim=50)(label_in)`**
  Converts the integer label (0 or 1) into a continuous 50-dimensional dense representation. This allows the model to learn a rich, continuous representation of class information.
* **`label_emb = Flatten()(label_emb)`**
  Flattens the embedding output from `(batch_size, 1, 50)` to `(batch_size, 50)` so it can be combined with the noise vector.

```python
    x = Concatenate()([noise_in, label_emb])
```
* Concatenates the 100-dimensional noise vector with the 50-dimensional label embedding. This yields a single 150-dimensional vector. This combination allows the generator to create class-specific EEG trials depending on the input label.

```python
    x = Dense(125 * 64, activation='relu')(x)
    x = Reshape((125, 64))(x)
```
* **`Dense(125 * 64, activation='relu')(x)`**
  Projects the 150-dimensional concatenated vector into a large $8000$-dimensional space. It uses the ReLU activation function to introduce non-linearity.
* **`Reshape((125, 64))(x)`**
  Reshapes this flat vector into a 2D matrix of shape `(125, 64)`. This acts as our initial low-resolution representation: a sequence length of 125 with 64 feature maps/channels.

```python
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 250))(x)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 500))(x)
    x = Conv1D(32, kernel_size=7, padding='same', activation='relu')(x)
    
    x = Lambda(lambda t: resize_1d(t, 1001))(x)
    x_out = Conv1D(n_channels, kernel_size=7, padding='same')(x) # output is (1001, 10)
```
* **`Conv1D(64, kernel_size=7, ...)`**
  Processes features over time using a 1D convolution with a filter window of 7 timepoints.
* **`Lambda(lambda t: resize_1d(t, ...))`**
  Applies our custom upsampling helper to stretch the sequence length:
  * First upsampling: `125` $\rightarrow$ `250`
  * Second upsampling: `250` $\rightarrow$ `500`
  * Third upsampling: `500` $\rightarrow$ `1001`
* **`x_out = Conv1D(n_channels, kernel_size=7, padding='same')(x)`**
  The final layer uses 1D convolution to map the 32 filters down to the target 10 EEG channels.
* **Why no activation on the output layer?**
  EEG signals contain negative and positive amplitudes. The real data is normalized between $-1$ and $+1$ using a `MinMaxScaler`. A linear output layer allows the generator to output any range of continuous values, and the WGAN-GP training process constrains it to match the target normalized distribution.

```python
    return Model([noise_in, label_in], x_out, name=\"Generator\")
```
* Returns the compiled Keras model mapping `[noise, label]` inputs to the generated EEG signal.

---

## 4. The Critic Model: `make_critic`

The Critic evaluates an EEG signal (real or fake) under the context of a target class label and outputs a single scalar score representing how "real" the sample is.

```python
def make_critic(n_channels=10):
    eeg_in = Input(shape=(1001, n_channels))
    label_in = Input(shape=(1,), dtype='int32')
```
* **`eeg_in = Input(shape=(1001, n_channels))`**
  Defines the input layer for the EEG signal of shape `(1001, 10)`.
* **`label_in = Input(shape=(1,), dtype='int32')`**
  Defines the input layer for the target class label (0 or 1).

```python
    label_emb = Embedding(input_dim=2, output_dim=50)(label_in)
    label_emb = Dense(1001)(label_emb)
    label_emb = Reshape((1001, 1))(label_emb)
```
* **`Embedding(input_dim=2, output_dim=50)(label_in)`**
  Embeds the class label into a 50-dimensional vector.
* **`Dense(1001)(label_emb)`**
  Projects the 50-dimensional embedding to 1001 dimensions to match the sequence length of the EEG signal.
* **`Reshape((1001, 1))(label_emb)`**
  Reshapes the vector into shape `(1001, 1)` representing a single channel.

```python
    x = Concatenate(axis=-1)([eeg_in, label_emb])
```
* Concatenates the embedded class label with the input EEG signal along the channel axis. This results in a tensor of shape `(1001, 11)` (10 EEG channels + 1 label channel).
* **Why do this?**
  This forces the Critic to not only grade if the EEG signal looks realistic in general, but specifically if its features align with the requested class (e.g. Mu/Beta desynchronization occurring in the correct motor hemisphere for a left/right hand trial).

```python
    x = Conv1D(32, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(128, kernel_size=7, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
```
* **`Conv1D(..., strides=2)`**
  Uses strided convolutions instead of pooling layers (like MaxPooling) to progressively downsample the signal.
  * Strided Conv 1: downsamples sequence length from `1001` $\rightarrow$ `501`
  * Strided Conv 2: downsamples sequence length from `501` $\rightarrow$ `251`
  * Strided Conv 3: downsamples sequence length from `251` $\rightarrow$ `126`
* **`LeakyReLU(0.2)(x)`**
  Activation function that allows a small, non-zero gradient ($20\%$ of the input value) when the unit is inactive ($x < 0$). This prevents "dead neurons" where network components stop learning.

```python
    x = Flatten()(x)
    x_out = Dense(1)(x) # linear output for WGAN (again, absolutely no sigmoid here!)
    
    return Model([eeg_in, label_in], x_out, name=\"Critic\")
```
* **`x = Flatten()(x)`**
  Flattens the final downsampled feature maps of shape `(126, 128)` into a 1D vector of size `16,128`.
* **`Dense(1)(x)`**
  Outputs a single, unbounded real number.
* **Why no Sigmoid?**
  In standard GANs, the discriminator uses a `sigmoid` activation to output a probability between 0 (fake) and 1 (real). WGAN-GP computes the Wasserstein distance, which does not represent a probability. It is a distance metric, and therefore the Critic must output raw, unbounded scores.

---

## 5. The Training Loop: `train_wgan_gp_for_subject`

This is the custom training loop that orchestrates WGAN-GP optimization, handling the mathematical details of Wasserstein loss and gradient penalty.

```python
def train_wgan_gp_for_subject(X_real_data, y_train_data, epochs=100, verbose=False, n_channels=10):
    X_real_data = X_real_data.astype(np.float32)
    X_real_gan = np.transpose(X_real_data[..., 0], (0, 2, 1))
```
* **`X_real_data.astype(np.float32)`**
  Casts the real training data to 32-bit floats for GPU compatibility.
* **`np.transpose(X_real_data[..., 0], (0, 2, 1))`**
  The preprocessed data from Phase 1 is stored in the shape `(batch, channels, timepoints, 1)` where the last dimension is a dummy dimension for 2D convolutions. 
  This line:
  1. Drops the dummy dimension: `X_real_data[..., 0]` yielding shape `(batch, channels, timepoints)`.
  2. Transposes the axes to `(batch, timepoints, channels)` to match the expected format for the generator and critic's 1D layers (`(1001, 10)`).

```python
    generator = make_generator(n_channels=n_channels)
    critic = make_critic(n_channels=n_channels)
```
* Instantiates the Generator and Critic networks.

```python
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
```
* Sets up the Adam optimizers. The hyperparameters are carefully chosen based on recommendations from the WGAN-GP paper:
  * `learning_rate=1e-4`: A slightly lower learning rate to maintain stability.
  * `beta_1=0.5`: Reduced momentum to prevent the optimization from oscillating wildly.
  * `beta_2=0.9`: Decay rate of second-order momentum.

### The Compiled `train_step` Function
```python
    @tf.function
    def train_step(real_eeg, labels):
```
* **`@tf.function`**
  Compiles the function into a static TensorFlow graph. This speeds up training significantly by optimizing GPU kernel launches.

#### Step 1: Train the Critic `n_critic` times
```python
        for _ in range(n_critic):
            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as critic_tape:
                fake_eeg = generator([noise, labels], training=True)
                d_real = critic([real_eeg, labels], training=True)
                d_fake = critic([fake_eeg, labels], training=True)
```
* **`for _ in range(n_critic):`**
  WGAN requires the Critic to remain close to optimal throughout training. Thus, the Critic is trained `n_critic = 5` times for every single update of the Generator.
* **`noise = tf.random.normal(...)`**
  Generates a batch of random normal vectors to feed into the generator.
* **`with tf.GradientTape() as critic_tape:`**
  Opens a context recorder that tracks all mathematical operations to compute gradients for the Critic's parameters automatically.
* **`fake_eeg = generator(...)`**
  The generator synthesizes a batch of fake EEG trials.
* **`d_real` & `d_fake`**
  The Critic scores the real batch and the generated fake batch.

#### Step 2: The Gradient Penalty Math
To satisfy the 1-Lipschitz condition, the norm of the Critic's gradients with respect to its inputs must not exceed 1. WGAN-GP achieves this by penalizing the model if the gradient norm deviates from 1.

```python
                epsilon = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
                interpolated = epsilon * real_eeg + (1 - epsilon) * fake_eeg
```
* **`epsilon`**
  Generates a vector of random mixing factors between 0 and 1 for each sample in the batch.
* **`interpolated`**
  Creates synthetic samples that lie along straight lines connecting real samples and generated fake samples. This ensures the Lipschitz constraint is checked throughout the input space.

```python
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    d_hat = critic([interpolated, labels], training=True)
```
* **`gp_tape.watch(interpolated)`**
  Tells TensorFlow to explicitly track gradients with respect to the `interpolated` samples themselves (which are tensor values, not model variables).
* **`d_hat`**
  The Critic scores the interpolated samples.

```python
                gradients = gp_tape.gradient(d_hat, interpolated)
                norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-12)
                gp = tf.reduce_mean((norms - 1.0) ** 2)
```
* **`gradients`**
  Calculates the gradient of the Critic's scores ($d\_hat$) with respect to the input samples ($interpolated$). This yields a vector of shape `(batch_size, 1001, 10)`.
* **`norms`**
  Computes the L2 norm (Euclidean length) of the gradient vector for each sample in the batch. A small constant (`1e-12`) is added inside the square root to prevent division by zero during backpropagation.
* **`gp`**
  Calculates the mean squared error between the gradient norms and the target value of `1.0`.

#### Step 3: Compute Critic Loss and Update
```python
                critic_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + gp_weight * gp
```
* **`critic_loss`**
  Computes the loss function of the Critic:
  $$\text{Critic Loss} = \mathbb{E}[D(\tilde{x})] - \mathbb{E}[D(x)] + \lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$
  Where:
  * $\mathbb{E}[D(\tilde{x})]$ is `tf.reduce_mean(d_fake)`: The Critic wants to assign a *low* score to fake data.
  * $\mathbb{E}[D(x)]$ is `tf.reduce_mean(d_real)`: The Critic wants to assign a *high* score to real data. Minimizing $-\mathbb{E}[D(x)]$ maximizes the score for real data.
  * $\lambda \cdot GP$ is `gp_weight * gp` (where $\lambda = 10.0$): Adds the penalty constraint.

```python
            gradients_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(gradients_critic, critic.trainable_variables))
```
* Backpropagates the loss and updates the weights of the Critic network.

#### Step 4: Train the Generator
After the Critic has been updated 5 times, we update the Generator once.
```python
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape:
            fake_eeg = generator([noise, labels], training=True)
            d_fake = critic([fake_eeg, labels], training=True)
            gen_loss = -tf.reduce_mean(d_fake)
```
* **`gen_loss = -tf.reduce_mean(d_fake)`**
  The Generator wants the Critic to believe its fake data is real. Therefore, the generator wants to maximize $D(\tilde{x})$. To maximize this expectation, we minimize its negative: `-tf.reduce_mean(d_fake)`.

```python
        gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
        return critic_loss, gen_loss
```
* Calculates the gradients of the Generator's parameters and updates the Generator network's weights.

---

### The Epoch Loop
```python
    dataset = tf.data.Dataset.from_tensor_slices((X_real_gan, y_train_data)).shuffle(len(y_train_data)).batch(batch_size, drop_remainder=True)
```
* Creates a TensorFlow input pipeline:
  * **`from_tensor_slices`**: Feeds the real transposed EEG data and class labels together.
  * **`shuffle`**: Randomly shuffles the data order at the start of each epoch.
  * **`batch(batch_size, drop_remainder=True)`**: Organizes data into batches of size 64. `drop_remainder=True` is critical because the gradient penalty logic calculates norms over a static tensor size, so variable batch sizes at the very end of an epoch would cause code errors.

```python
    for epoch in range(epochs):
        for real_batch, labels_batch in dataset:
            labels_batch = tf.expand_dims(labels_batch, axis=-1)
            train_step(real_batch, labels_batch)
            
    return generator, critic
```
* Iterates over the requested number of training epochs, running the `train_step` on each batch.
* Returns the trained Generator and Critic models.

---

## 6. Why WGAN-GP for EEG Augmentation?

EEG signals have a very low Signal-to-Noise Ratio (SNR) and are notoriously difficult for standard neural networks to generate because they are highly non-stationary and subject-dependent.

Using **WGAN-GP** solves these issues:
1. **Mathematical Stability:** The Wasserstein loss landscape provides continuous, smooth gradients even when the Generator and Critic are not perfectly matched in capability.
2. **Avoiding Mode Collapse:** The Gradient Penalty constraint forces the Critic to learn smooth, continuous scoring fields. This prevents the Generator from finding a single "cheat waveform" and repeating it; it must learn to generate a diverse range of realistic waveforms that map to the underlying real distributions.
3. **Quality Filtering (Critic Scoring):** Because the Critic outputs a continuous "realness" score, it acts as a built-in quality control mechanism. Later in the code, the pipeline generates a large pool of synthetic samples and uses this Critic to filter out the lower-quality generations, keeping only the top 25% of samples that scoring-wise look closest to real trials.
