

# CS5760 Natural Language Processing - Homework 4
**Student Name:** Rishik Vardhan Reddy Ummenthala
**Student ID: 700777149

## Project Overview
This repository contains the implementation of several foundational Natural Language Processing (NLP) architectures using PyTorch. The assignment covers Feedforward Neural Networks, XOR ReLU networks, and advanced sequence models including RNNs and Transformers.

### Repository Structure
* `NLP_HW4.ipynb`: The primary Jupyter Notebook containing all mathematical derivations and programming implementations.
* `README.md`: This file, providing student info and work explanation. 

---

## Part I: Mathematical & Short Answer
This section includes manual calculations and conceptual explanations for:
* Multi-Input Feedforward Networks:** Manual computation of pre-activations, sigmoid activations, and Binary Cross-Entropy loss.
* XOR with ReLU Network:** Extending the XOR network to 3 hidden units and analyzing the decision boundary. 
* Perceptron Decision Boundaries:** Sketching boundaries and performing manual weight updates using a learning rate ($\eta = 0.5$). 
* Advanced NLP Concepts:** Explanations of LSTM gates, Vanishing Gradients, and the Self-Attention mechanism.

---

## Part II: Programming Implementations

### Q1. Character-Level RNN Language Model
* Goal:** Predict the next character in a sequence using a toy corpus. 
* Architecture:** Embedding → GRU → Linear → Softmax. 
* Training:** Implemented using **Teacher Forcing** and the Adam optimizer. 
* Generation:** Includes a sampling loop with **Temperature Control** ($\tau = 0.7, 1.0, 1.2$) to vary the randomness of output text. 

### Q2. Mini Transformer Encoder
* Goal:** Process sentences to generate contextual embeddings.
* **Components:** * Sinusoidal Positional Encoding to provide sequence order. 
    * Multi-Head Attention (2 heads) for parallel context processing.
    * Feed-Forward layers with ReLU activation. 
    * **Add & Norm** blocks (Residual connections + Layer Normalization). 
* **Visualization:** Includes an **Attention Heatmap** demonstrating how the model weights relations between words like "love" and "nlp". 

### Q3. Scaled Dot-Product Attention
* **Goal:** Standalone implementation of the Transformer attention formula: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
* **Stability Check:** Demonstrated how the scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the Softmax distribution from becoming too "peaky," which mitigates vanishing gradients during training. 

---

## Reflections 
* **Sequence Length & Hidden Size:** Increasing the hidden size (e.g., from 64 to 256) significantly improves the model's ability to memorize the toy corpus, but requires more training time to avoid overfitting.
* **Temperature Impact:** * **Low Temperature (0.7):** Produces very conservative, repetitive text that stays close to the training data.
    * **High Temperature (1.2):** Increases "creativity" but leads to frequent misspellings as the model samples from lower-probability character distributions.
* **Teacher Forcing Trade-off:** Using teacher forcing during training speeds up convergence significantly, but the model sometimes struggles during generation (inference) because it is not used to seeing its own mistakes.

**Comments:** Code is documented with comments explaining the mathematical logic and layer structure. 
