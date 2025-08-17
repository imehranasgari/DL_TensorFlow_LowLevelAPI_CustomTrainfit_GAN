# Customizing TensorFlow's `model.fit()` and Implementing a GAN

## Problem Statement and Goal of the Project

This project explores advanced functionalities within TensorFlow and Keras. The goal is twofold:

1.  To gain granular control over the training process by overriding the default `train_step` method of the `tf.keras.Model` class.
2.  To apply this low-level control to implement a Generative Adversarial Network (GAN) from scratch, showcasing a deep understanding of model architecture and training loops.

This demonstrates a move beyond high-level APIs to a more fundamental understanding of how models are trained.

-----

## Solution Approach

### Custom Training Loop

To take control of the training process, I subclassed the `tf.keras.Model`. By overriding the `train_step` function, I manually defined the sequence of operations for each training batch:

  * **Forward Pass**: The model's predictions are generated.
  * **Loss Calculation**: The loss is computed based on the predictions and true labels.
  * **Gradient Computation**: `tf.GradientTape` is used to record operations and calculate gradients.
  * **Weight Updates**: The optimizer applies the computed gradients to update the model's weights.
  * **Metric Updates**: Metrics such as Mean Absolute Error (MAE) are updated.

This was demonstrated in three stages:

1.  **High-Level Override**: Using built-in loss and metrics functions.
2.  **Low-Level Override**: Manually calculating the loss and updating metrics.
3.  **Weight Support**: Extending the `train_step` to handle `sample_weight` and `class_weight` for imbalanced datasets.

### Generative Adversarial Network (GAN)

A GAN was implemented to generate 28x28 grayscale images. The network consists of two main components:

  * **Generator**: Takes a 128-dimensional random noise vector as input and upsamples it using `Conv2DTranspose` layers to produce a 28x28x1 image.
  * **Discriminator**: A standard convolutional network that takes a 28x28x1 image as input and classifies it as either "real" or "fake".

-----

## Technologies & Libraries

  * **TensorFlow**
  * **Keras**
  * **NumPy**

-----

## Description about Dataset

For the custom `train_step` demonstrations, a synthetic dataset was created using `np.random.random`. This was sufficient to verify that the custom training loop was functioning correctly without the need for a specific dataset.

The GAN is designed to work with datasets of 28x28x1 images, such as the MNIST handwritten digit dataset.

-----

## Installation & Execution Guide

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install tensorflow numpy
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook customizing_what_happens_in_fit_me.ipynb
    ```

-----

## Key Results / Performance

This project was focused on implementation and understanding rather than achieving high performance metrics.

  * The custom `train_step` models successfully trained, with the loss decreasing over epochs, confirming the correctness of the custom training loops.
  * The GAN's **Generator** and **Discriminator** models were successfully built. Their architecture summaries are provided below, showing the layer configurations and parameter counts.

#### Generator Architecture

```
Model: "generator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                 │ (None, 6272)           │       809,088 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_7 (LeakyReLU)       │ (None, 6272)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ reshape_1 (Reshape)             │ (None, 7, 7, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_2              │ (None, 14, 14, 128)    │       262,272 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_8 (LeakyReLU)       │ (None, 14, 14, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_3              │ (None, 28, 28, 128)    │       262,272 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_9 (LeakyReLU)       │ (None, 28, 28, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 28, 28, 1)      │         6,273 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,339,905 (5.11 MB)
 Trainable params: 1,339,905 (5.11 MB)
 Non-trainable params: 0 (0.00 B)
```

#### Discriminator Architecture

```
Model: "discriminator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_3 (Conv2D)               │ (None, 14, 14, 64)     │           640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_5 (LeakyReLU)       │ (None, 14, 14, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 7, 7, 128)      │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_6 (LeakyReLU)       │ (None, 7, 7, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling2d_1          │ (None, 128)            │             0 │
│ (GlobalMaxPooling2D)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 74,625 (291.50 KB)
 Trainable params: 74,625 (291.50 KB)
 Non-trainable params: 0 (0.00 B)
```

-----

## Screenshots / Sample Output

Not provided.

-----

## Additional Learnings / Reflections

This project provided valuable hands-on experience with the lower-level aspects of model training in TensorFlow.

  * **Deeper Framework Understanding**: Overriding `train_step` demystified what happens behind the scenes during `model.fit()`. It clarified the roles of the forward pass, gradient taping, and optimizer application in a tangible way.
  * **Flexibility and Control**: I now have a clear understanding of how to implement unconventional training procedures, such as those required for GANs, reinforcement learning, or models with multiple optimizers and losses.
  * **GAN Implementation**: Building a GAN from scratch reinforced my knowledge of generator and discriminator architectures, the importance of activation functions like `LeakyReLU` in preventing issues like dying ReLUs, and the use of transposed convolutions for upsampling.

This deeper knowledge is crucial for developing custom solutions and debugging complex models, moving beyond off-the-shelf implementations.

-----

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## 👤 Author

## Mehran Asgari

## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)

## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.