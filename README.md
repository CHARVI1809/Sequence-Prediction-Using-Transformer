# 🚀 Sequence Prediction using a Decoder‑Only Transformer

## 📖 Overview
This project implements a **decoder-only Transformer from scratch using PyTorch** to perform **next-token prediction** on synthetic robotic trajectory data.

The model learns to predict the **next state/action token** in a sequence, demonstrating the idea that **robot control can be framed as sequence modeling**, similar to language modeling.

This project is based on the concept of the **Unified Sequence Hypothesis**, where different modalities (text, actions, states) are treated as token sequences.

---

## 🧠 Key Idea

> **Sequence Prediction = Predict the next token using past tokens**

- Input: sequence of robot states/actions  
- Output: next token in the sequence  
- Training: autoregressive (teacher forcing)

---

## 🤖 Why Transformers?

Traditional models (RNNs/LSTMs):
- ❌ Struggle with long-range dependencies  
- ❌ Sequential computation (slow)  
- ❌ Gradient instability  

Transformers solve this using:
- ✅ Self-Attention (captures global context)  
- ✅ Parallel computation  
- ✅ Better scalability  
- ✅ Stable training with normalization  

---

### ⚠️ Important Note
- ❌ No use of `nn.Transformer` or pre-built models  
- ✅ All components implemented manually  

---

## 📊 Dataset: Synthetic Robotic Trajectories

- **Samples:** 10,000 trajectories  
- **Sequence Length:** 50  
- **State Representation:**  
  - 7-DOF joint angles  
  - 3D end-effector position  
- **Tokenization:** Discretized into 256 bins  
- **Vocabulary Size:** 256  

## 🏗️ Model Architecture

The architecture is based on:

> **"Attention Is All You Need" (Vaswani et al., 2017)**

### 🔹 Components
- Embedding Layer  
- Positional Encoding  
- Multi-Head Self Attention  
- Feed Forward Network  
- Layer Normalization  
- Dropout  
- Output Projection Layer  

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Framework:** PyTorch  
- **Libraries:** NumPy, Matplotlib, Scikit-learn  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 📊 Training Details

- Loss Function: Cross-Entropy / MSE (task dependent)
- Optimizer: Adam
- Learning Rate: Configurable
- Batch Size: Configurable
- Epochs: Configurable

 ## 👩‍💻 Authors

**Charvi Gupta**  
📍 B.Tech CSE, Manipal University Jaipur  

**Shreeya Sahai**  
📍 B.Tech CSE, Manipal University Jaipur
