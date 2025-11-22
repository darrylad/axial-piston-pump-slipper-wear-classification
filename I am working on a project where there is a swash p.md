<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I am working on a project where there is a swash plate axial piston pump, and using the x, y, z vs time acceleration (vibration) data from an accelerometer attached to the pump, I need to predict the amount of wear in the pisons. I have attached some related papers. there are 5 classes this way.

.
├── 10 um
│   └── 10 um_merged.csv
├── 100 um
│   └── 100 um_merged.csv
├── 45 um
│   └── 45 um_merged.csv
├── 75 um
│   └── 75 um_merged.csv
├── Healthy
│   └── Healthy_merged.csv

I have split these merged csv files into 700 smaller csv files. I have attached a csv file for your reference. Channel name is time, x, y and z are vibrations (accelerations) in x, y, and z directions respectively. I need to train a model, such that from a vibration data, it can classify it into one of these 5 classes. What is the best approach I must take to do this? I am new to machine learning, so can you please explain in simple terms systematically how we can do this, and why? what will be novel in this method?

A practical, high-accuracy path is to convert each tri-axial vibration window into a time–frequency image and train a light CNN with batch normalization whose hyperparameters are tuned automatically (Bayesian optimization or PSO), optionally adding a 1D temporal branch (LSTM) or a small Siamese/graph module to boost small-sample performance and multi-axis fusion fidelity. This mirrors what works best on axial piston pumps in recent studies while adapting it to your x–y–z accelerometer data and five wear classes.[^1_1][^1_2][^1_3][^1_4][^1_5]

### Your data

The sample you shared has columns named time, X, Y, Z, so each record is a tri-axial acceleration snapshot at a given timestamp.  The filename indicates one wear class (10-um), so your 700-window split per class can be treated as labeled samples for a five-class classifier.[^1_6]

### Why time–frequency first

Axial piston pump vibration is strongly nonstationary; continuous wavelet transforms (CWT) or synchrosqueezed wavelet transforms (SWT) expose fault-related signatures across time and frequency that are harder to separate in raw time series.  On piston pump rigs comparable to yours, CWT/SWT images of vibration/pressure/acoustic signals enable CNNs to cleanly separate normal, swash plate wear, slipper wear, loose slipper, and central spring faults at high accuracy.[^1_2][^1_4][^1_1]

### Recommended model

Use a light “normalized CNN” (CNN+BatchNorm) on the 2D CWT images formed by stacking X, Y, and Z as channels, and let Bayesian Optimization select learning rate, batch size, kernel sizes and counts, and FC sizes; this combination has achieved 97–100% accuracy on piston pumps while mitigating overfitting and optimizer sensitivity.  If you prefer metaheuristics, PSO-tuned improved LeNet-5 has reached ~99% on five piston-pump classes using CWT inputs and is a strong reference baseline.[^1_1][^1_2]

### Add temporal context (optional)

To capture longer-range dynamics across windows, add a parallel 1D branch: a compact 1D-CNN over raw x–y–z sequences feeding a small LSTM before late fusion with the 2D CNN logits; LSTM fusion has been shown effective for pump fault diagnosis when time dependencies matter.  This hybrid “2D-CWT-CNN + 1D-CNN-LSTM” often improves robustness when the machine load or noise varies.[^1_5]

### Small-data and multi-axis fusion boost (optional)

When labeled data are limited, a Siamese-style auxiliary objective that pulls together embeddings from same-class windows and pushes apart different classes improves class separation; recent piston-pump work extends this with a random spatiotemporal graph over multi-sensor channels to better exploit correlations, which you can simplify to your three axes.  Concretely, add a Siamese contrastive loss on pairs of x–y–z windows and train jointly with cross-entropy, which stabilizes learning in five-way severity classification with few samples.[^1_3]

### End-to-end pipeline

- Segment windows: use your existing 700 sub-CSV windows per class; verify step timing from the time column and retain synchronized X–Y–Z segments.[^1_6]
- Transform: compute per-axis CWT (Morlet), stack axes as a 3-channel image; SWT is an alternative that further sharpens ridges in time–frequency space.[^1_4][^1_1]
- Normalize: per-axis standardization plus BatchNorm in the network improves stability and permits higher learning rates.[^1_2]
- Train: start with NCNN (2 conv blocks + BN + MaxPool + 2 FC + Softmax), and let Bayesian Optimization choose LR, batch size, kernels, and FC sizes; alternatively, use PSO to tune an improved LeNet-5.[^1_1][^1_2]
- Validate: track per-class accuracy and confusion matrices; piston-pump studies consistently analyze confusion to spot confusions like loose-slipper vs spring failure.[^1_2][^1_1]
- Visualize: use t-SNE on penultimate-layer embeddings to verify five well-separated clusters as accuracy rises.[^1_1]


### Concrete hyperparameters to try

Start with 64×64 or 128×128 CWT images; two conv blocks with 3×3 and 5×5 kernels, 32–64 filters per block, global average pooling, and 128–256-neuron FC before Softmax often suffice for high accuracy when paired with BN.  Set BO/PSO search ranges similar to those reported for piston pumps (e.g., LR ~1e−3–1e−2, batch size 24–56, modest kernel counts), which converged quickly to >98% accuracy in multiple signals.[^1_2][^1_1]

### Baselines to compare

- CWT + Improved LeNet-5 (PSO tuned): strong accuracy/latency trade-off on five states.[^1_1]
- NCNN (BN + BO) on vibration CWT: high accuracy and stable training with few iterations.[^1_2]
- Hybrid VGG-LSTM on time–frequency features: boosts when longer temporal context helps.[^1_5]

| Approach | Strength | Note |
| :-- | :-- | :-- |
| CWT + PSO-LeNet | High accuracy with fast inference [^1_1] | Good starting baseline [^1_1] |
| NCNN + BO | Stable, near-SOTA accuracy with BN [^1_2] | Fewer layers, easy to tune [^1_2] |
| Hybrid CNN+LSTM | Captures long-term temporal cues [^1_5] | Helpful under variable loads [^1_5] |
| Siamese/graph add-on | Better small-sample generalization [^1_3] | Pair with cross-entropy (CCE) [^1_3] |

### Why this works

Time–frequency images make nonstationary wear signatures linearly separable for shallow CNNs, matching repeated successes on piston pumps and other rotating machinery.  Batch normalization reduces internal covariate shift and regularizes training, while BO/PSO remove manual hyperparameter guesswork that often limits newcomers.  Where available data are limited or axes carry complementary cues, a small Siamese/graph or LSTM branch can markedly lift performance without large model size.[^1_3][^1_5][^1_1][^1_2]

### What is novel here

- Tailored multi-branch fusion for tri-axial accelerometer: fusing a per-axis CWT-CNN with a light 1D CNN–LSTM branch yields a compact, piston-pump–specific model that captures both instantaneous spectral patterns and cross-window dynamics, which extends published NCNN and LSTM ideas to a unified architecture for your x–y–z data.[^1_5][^1_2]
- Contrastive-assisted training for five severity classes: adding a Siamese contrastive term during training (without changing the classifier at inference) adapts recent small-sample piston-pump advances to tri-axial vibration classification, improving separability of adjacent wear levels.[^1_3]
- Automated, light model selection: combining BN with BO or PSO to find the smallest network that meets your accuracy target reduces overfitting risk and training time compared with manual tuning-heavy pipelines in earlier work.[^1_1][^1_2]


### Practical tips

Ensure class-balanced sampling per wear level and stratified splits by run/session to avoid leakage across windows from the same continuous acquisition.  Use confusion matrices to watch for systematic confusions (e.g., loose slipper vs spring failure) and, if seen, increase temporal context (LSTM) or add contrastive pairs spanning those classes.[^1_5][^1_2][^1_1]

### Minimum viable plan (2–3 weeks)

- Week 1: generate per-window CWT images from X–Y–Z, standardize axes, assemble train/val/test splits from your 700-window-per-class sets.[^1_6][^1_1]
- Week 2: train NCNN with BO; compare to PSO-LeNet; pick the lighter model that crosses your accuracy bar on validation.[^1_2][^1_1]
- Week 3: if needed, add the 1D CNN–LSTM branch or the Siamese contrastive head; re-tune LR/batch with BO and finalize with confusion analysis and t-SNE.[^1_3][^1_5][^1_1]
<span style="display:none">[^1_7][^1_8]</span>

<div align="center">⁂</div>

[^1_1]: 2021-Intelligent-fault-diagnosis-of-hydraulic-piston-pump-combining-improved.pdf

[^1_2]: 2024-A-light-deep-adaptive-framework-toward-fault-diagnosis-of-a-hydraulic.pdf

[^1_3]: 2025-SRSGCN-A-novel-multi-sensor-fault-diagnosis-method-for-hydraulic-axial.pdf

[^1_4]: 2022-Intelligent-fault-identification-of-hydraulic-pump-using-deep-adaptive.pdf

[^1_5]: 2025-A-fault-identification-method-of-hydraulic-pump-fusing-long-short-term.pdf

[^1_6]: 10-um_merged_cleaned_20251112_190203_1.csv

[^1_7]: 2023-Failure-Analysis-and-Intelligent-Identification-of-Critical.pdf

[^1_8]: 2022-An-adaptive-deep-learning-model-towards-fault-diagnosis-of-hydraulic-piston-pump-using-pres.pdf


---

# Systematic Approach to CWT for Axial Piston Pump Vibration Classification

Let's break down the **Continuous Wavelet Transform (CWT)** step-by-step for your fault diagnosis task, focusing on why and how to apply it to your tri-axial (x, y, z) accelerometer data.

## 1. **Why use CWT?**

- **Vibration signals are non-stationary:** Machine faults often produce transient, time-varying features that are difficult to capture with traditional Fourier methods.[^2_1][^2_12]
- **CWT provides time–frequency analysis:** It decomposes your vibration data into components at multiple frequencies and times, allowing you to see how frequency content changes over time.
- **Scalogram images are perfect for CNN input:** The CWT output (scalogram) is a 2D image showing how the signal's energy is distributed across time and frequency, and these are highly effective as input for deep learning models.[^2_12][^2_13]


## 2. **Why the Morlet wavelet?**

- **Morlet is widely used for machinery vibration:** It provides a great balance between time and frequency localization, and is particularly effective for signals with oscillatory characteristics.[^2_3][^2_8][^2_12]
- **Complex Morlet is optimal for spectral sharpness:** Studies show high classification accuracy in rotating machinery when using complex Morlet for CWT scalograms.[^2_8]
- **Parameter selection:** The main parameter is the wavelet's bandwidth (often denoted $\sigma$ or $\omega_0$), which controls trade-off between time and frequency precision. Typical $\omega_0$ values are between 5–8; $\sigma = 6$ is a common choice for clear defect visualization.[^2_3]


## 3. **Handling x, y, z channels**

- **Process each axis separately:** For each windowed vibration sample (let's say, 2–5 seconds), calculate the CWT (scalogram) for the x, y, and z channels independently.
- **Create multichannel images:** Stack the three scalograms to form a RGB image (each channel corresponding to x, y, and z)—this is standard practice, allowing the CNN to use spatial and cross-axis features.
    - *Alternative:* If you want to process channels independently, train three parallel models and fuse their outputs, but joint-channel images usually work better for vibration problems.


## 4. **Step-by-Step Implementation**

### A. **Choose window length and overlap**

- Use a short time window (1–5 seconds of data), with overlap if needed (commonly 50%, but can vary).
- Ensure that each window corresponds to one example with a known class label.


### B. **Apply CWT to each axis**

- For each window:
    - For each channel (x, y, z):
        - Use a CWT library (Python: PyWavelets, Scipy, Matlab's cwt, or fast implementations like “fCWT”)[^2_3]
        - Set scales (frequency range) based on your sampling rate and relevant mechanical frequencies
        - Use the complex Morlet wavelet for maximal defect separation
        - Normalize the scalogram (e.g., min-max or z-score)


### C. **Form the final multi-channel image**

- Stack the three scalograms into a single image of shape [frequency, time, 3]
- Save as PNG/JPG, or keep in memory as numpy arrays/tensors if training models


### D. **Why scalograms?**

- **Scalograms = visual maps:** They show how vibration energy varies over both time and frequency, highlighting fault-induced shifts.[^2_13][^2_12]
- **Suitable for CNNs:** Convolutional neural networks excel at finding patterns in images. By turning vibration data into scalograms, you let the model 'see' fault-related behaviors that are hard to extract using manual features.


## 5. **Summary Workflow**

1. **Window** your raw data (1–5 s segments, optionally overlapping)
2. **Compute CWT** with Morlet wavelet for each channel (x, y, z)
3. **Stack scalograms** to form 3-channel images
4. **Label and store** each image according to its wear class
5. **Train your CNN** using these time–frequency images as inputs

***
### **Mini-review:**

- **CWT is crucial for fault detection in rotating machinery** because it reveals localized changes at frequencies tied to wear, impacts, or looseness that are obscured in spectral averages.
- **Morlet is chosen for its frequency sharpness and oscillatory nature**, aligning closely with vibration patterns.
- **Stacked 'scalogram images' join multi-axis information**, exactly matching best practice for vibration-driven machine learning.

If you want more detail on the CWT code or parameter selection, just ask! And, if you share your project specifics (course level or background), I can tailor the explanation even more.
<span style="display:none">[^2_10][^2_11][^2_14][^2_2][^2_4][^2_5][^2_6][^2_7][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://www.electricajournal.org/Content/files/sayilar/62/581-584.pdf

[^2_2]: https://www.extrica.com/article/19611

[^2_3]: https://www.nature.com/articles/s43588-021-00183-z

[^2_4]: http://jtam.pl/pdf-101997-33558?filename=Wavelets+and+principal.pdf

[^2_5]: https://rpubs.com/atrujill/715279

[^2_6]: https://ieeexplore.ieee.org/document/6292787

[^2_7]: https://www.matec-conferences.org/articles/matecconf/pdf/2016/42/matecconf_imet2016_01061.pdf

[^2_8]: https://www.mdpi.com/2079-9292/13/2/452/pdf?version=1705905842

[^2_9]: https://www.mdpi.com/1424-8220/23/10/4781

[^2_10]: http://www.jatm.com.br/ojs/index.php/jatm/article/view/804

[^2_11]: https://downloads.hindawi.com/journals/mpe/2021/6626232.pdf

[^2_12]: https://www.mdpi.com/1424-8220/18/5/1429/pdf

[^2_13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5982639/

[^2_14]: https://www.mdpi.com/1424-8220/22/22/8760/pdf?version=1668262893


---

# I have done cwt on them, and i have the images now, in rgb, for x, y, z. Now, how can we proceed to the training? can you please teach me systematically?

Now that you have RGB CWT images for each vibration window, the next steps are to 1) structure the dataset, 2) define and implement a CNN, 3) train it with a suitable loss/optimizer, and 4) evaluate it with the right metrics.[^3_1][^3_2][^3_3]

***

### 1. Clarify what one sample is

- In your setup, **one vibration window** (from time series $x(t),y(t),z(t)$) has already been converted into **one RGB scalogram image**, where the three channels correspond to the three axes.[^3_2][^3_3][^3_1]
- Each such image should have a **single label**: Healthy, 10 µm, 45 µm, 75 µm, or 100 µm wear, just like the five states in the pump papers (normal, swash plate wear, etc.).[^3_3][^3_1][^3_2]

***

### 2. Organize your image dataset

A clean folder structure makes training much easier (and lets you use standard libraries directly).[^3_1][^3_2][^3_3]

- Create a root folder, for example: `data_cwt_images/`.[^3_2][^3_1]
- Inside it, create **one subfolder per class**:
    - `Healthy/`
    - `10um/`
    - `45um/`
    - `75um/`
    - `100um/`
Each folder contains all CWT images (RGB) belonging to that class.[^3_3][^3_1][^3_2]
- Use a consistent image format (PNG or JPG) and sensible filenames, for example `10um_run1_win003.png`.[^3_1][^3_3]

This mirrors what is done in the hydraulic pump studies where thousands of CWT or SWT images are organized by fault category for CNN training.[^3_2][^3_3][^3_1]

***

### 3. Split into train / validation / test

The split is critical to avoid “cheating” (data leakage).[^3_3][^3_1][^3_2]

- **Goal:** About 70% training, 15% validation, 15% test is a good starting point.[^3_1][^3_2]
- **Important:** If multiple windows come from the **same continuous original signal**, keep all those windows **in the same split** (all in train, or all in validation, or all in test).[^3_3][^3_1]
    - This mimics what the pump papers do: they split at the sample level, not by cutting one sample into overlapping training and test segments.[^3_2][^3_1]
- Make sure all five classes have similar proportions in each split to avoid class imbalance.[^3_2][^3_3]

You can either write a small script that creates three lists of file paths (train/val/test) or physically copy images into `train/`, `val/`, and `test/` subdirectories under each class.[^3_1][^3_3][^3_2]

***

### 4. Choose a CNN architecture (simple and effective)

For a first model, use a **light CNN with batch normalization**, inspired by the NCNN / light adaptive frameworks in your attached papers.[^3_3][^3_1][^3_2]

A very reasonable starting architecture:

- **Input**: 3 × H × W CWT image (e.g. resized to 64×64 or 128×128).[^3_2][^3_3]

Block 1

- Conv2D: 32 filters, 3×3, stride 1, padding 1.[^3_3][^3_2]
- BatchNorm
- ReLU
- MaxPool 2×2

Block 2

- Conv2D: 64 filters, 3×3, stride 1, padding 1.[^3_2][^3_3]
- BatchNorm
- ReLU
- MaxPool 2×2

Block 3

- Conv2D: 128 filters, 3×3, stride 1, padding 1.[^3_3][^3_2]
- BatchNorm
- ReLU
- MaxPool 2×2

Classifier

- Global Average Pooling (over spatial dimensions), producing a 128‑dim vector.[^3_2][^3_3]
- Fully Connected (Dense): 128 → 64, ReLU, optional dropout 0.3.[^3_3][^3_2]
- Fully Connected: 64 → 5, Softmax.[^3_1][^3_2]

This is conceptually the same as the improved LeNet / NCNN structures used on hydraulic pump CWT/SWT images, but slightly simplified for easier implementation.[^3_1][^3_2][^3_3]

***

### 5. Prepare the images for training

Before feeding images to the network, you need consistent preprocessing.[^3_1][^3_2][^3_3]

1. **Resize**
    - Resize every image to a fixed size, e.g. 64×64 or 128×128.[^3_2][^3_3]
    - The pump papers commonly use 64×64 or 224×224; you can start with 64×64 to keep the model small.[^3_1][^3_3][^3_2]
2. **Normalize pixel values**
    - Convert to floating point and scale to  (divide by 255) or standardize to zero mean and unit variance per channel.[^3_3][^3_1][^3_2]
    - Batch normalization in the network will then help stabilize training further.[^3_2][^3_3]
3. **Data augmentation (careful)**
    - Mild **random shifts** or **small random crops** along time or frequency directions can improve robustness, similar to small variability in operating conditions.[^3_3][^3_2]
    - Avoid operations that destroy the physical meaning (e.g. 180° rotations, vertical flips arbitrarily changing frequency axis).[^3_1][^3_2]

Most modern ML frameworks (PyTorch, TensorFlow/Keras) can handle these operations on-the-fly during training.[^3_2][^3_3]

***

### 6. Define loss, optimizer, and training hyperparameters

Your problem is **5-class classification**, so the standard setting is:[^3_1][^3_3][^3_2]

- **Loss function**: Categorical cross-entropy (in PyTorch: `nn.CrossEntropyLoss`).[^3_1][^3_2]
- **Optimizer**: Adam is a robust default (papers compare Adam, Adagrad, Adadelta and find Adam works well for these pump models).[^3_3][^3_2][^3_1]
- **Initial hyperparameters**:
    - Learning rate $\approx 10^{-3}$.[^3_2][^3_3][^3_1]
    - Batch size in the range 24–56, depending on memory.[^3_3][^3_1][^3_2]
    - Epochs: start with 30–60, and use **early stopping** when validation accuracy stops improving.[^3_1][^3_2][^3_3]

The Jiangsu group uses Adam with batch sizes in this range and 30–60 epochs for both the improved LeNet and NCNN variants on hydraulic pump CWT/SWT images.[^3_2][^3_3][^3_1]

***

### 7. Training loop: what actually happens

Conceptually, the training loop does this repeatedly:[^3_3][^3_1][^3_2]

1. Sample a batch of images and labels from the training set.[^3_1][^3_2]
2. Pass images through the CNN to get predicted class probabilities.[^3_2][^3_1]
3. Compute the loss between predictions and true labels (cross-entropy).[^3_3][^3_1][^3_2]
4. Backpropagate gradients and update weights using the optimizer.[^3_1][^3_2]
5. After each epoch, evaluate on the validation set, and track validation accuracy and loss.[^3_2][^3_3][^3_1]

Here is a **minimal PyTorch-style sketch** to make the process concrete (you do not need to understand every line immediately; think of it as a map):[^3_3][^3_1][^3_2]

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1. Transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),                      # Converts to [C, H, W] float32 in [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # adjust if you like
                         std=[0.5, 0.5, 0.5])
])

# 2. Datasets assuming: root/train/class_name/*.png etc.
train_data = datasets.ImageFolder('data_cwt_images/train', transform=transform)
val_data   = datasets.ImageFolder('data_cwt_images/val',   transform=transform)
test_data  = datasets.ImageFolder('data_cwt_images/test',  transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

# 3. Define a simple CNN (very similar to what we discussed)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),     # global average pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=5)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop (simplified)
for epoch in range(40):  # e.g., 40 epochs
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # after each epoch: compute validation accuracy (not shown in detail here)
```

This is a direct analogue of what the pump papers implement with their improved LeNet‑5 and NCNN architectures on CWT/SWT images.[^3_1][^3_2][^3_3]

***

### 8. Evaluating the model

Simply reporting “accuracy” is not enough; you want to know how well the model distinguishes each wear level.[^3_2][^3_3][^3_1]

- **Overall accuracy** on the held-out **test set**.[^3_1][^3_2]
- **Confusion matrix**: a 5×5 table where rows are true classes and columns are predicted classes.[^3_2][^3_1]
    - In the hydraulic pump studies, confusion matrices clearly showed which faults (e.g. loose slipper vs spring failure) tended to be confused, guiding model refinement.[^3_3][^3_1][^3_2]
- **Per-class accuracy / recall**: fraction of correctly detected samples for each wear level.[^3_3][^3_1][^3_2]

If you see, for example, that 75 µm and 100 µm are often confused, that may signal that you need slightly longer time windows or a slightly deeper CNN to capture subtle differences.[^3_1][^3_2][^3_3]

***

### 9. Where you can add novelty at the training stage

Even if CWT + CNN is well-studied, your project can still introduce novel elements in how training is done.[^3_2][^3_3][^3_1]

- **Hyperparameter search for the light CNN**: instead of fixing learning rate, batch size, and kernel counts by hand, you can follow the papers and use **Bayesian optimization** (or PSO) to find the best hyperparameters for your specific pump and wear-level labels.[^3_3][^3_1][^3_2]
- **Tri-axial fusion**: by designing and comparing different fusion strategies (e.g. stacked RGB vs. channel-attention over x–y–z), you can show which structure best captures wear-severity information from tri-axial vibration for an axial piston pump.[^3_1][^3_2][^3_3]
- **Severity-focused evaluation**: most prior work focuses on fault type (which component), while your five classes are explicitly **wear severity levels**; careful analysis of confusion between neighboring levels is itself a useful and relatively unexplored angle.[^3_2][^3_3][^3_1]

***

If you tell which framework you are using (PyTorch, TensorFlow/Keras, or Matlab), a next step can be to walk through a complete, concrete script tailored to your directory structure and CWT images.[^3_3][^3_1][^3_2]

<div align="center">⁂</div>

[^3_1]: 2021-Intelligent-fault-diagnosis-of-hydraulic-piston-pump-combining-improved.pdf

[^3_2]: 2022-Intelligent-fault-identification-of-hydraulic-pump-using-deep-adaptive.pdf

[^3_3]: 2024-A-light-deep-adaptive-framework-toward-fault-diagnosis-of-a-hydraulic.pdf


---

# I have attached some sample images for you to see. I have also attached tree.txt which shows the folder structure. There are currently varying number of images in each class, the lowest in healthy. But each cwt image was made on data with equal number, (340) of data points (i have attached one csv file for you). Each image is 256 by 256 resolution. now that I have these images in this structure, how shall i proceed? Since healthy class has the least number of images, shall i reduce the number of images in the other classes as well to match it, or is it not required? Then, how can I train this? Can you please give me a complete code? Can we use tensorflow, is it well suited here?

You are already in a very good position: you have clean 256×256 RGB CWT images per window, organized by wear class, plus the original x–y–z windows of equal length (340 points).  TensorFlow/Keras is well suited here because it can 1) load this kind of directory-structured image data easily and 2) lets you build and train CNNs with a few lines of code.[^4_1][^4_2][^4_3][^4_4][^4_5][^4_6]

Below is a step‑by‑step plan and then a **complete TensorFlow script** you can adapt.

***

### 1. What to do about class imbalance

From `tree.txt`, each wear level (e.g. “10 um”, “45 um”, “75 um”, “100 um”) has many images, while **Healthy** has noticeably fewer samples.  This is a classic **imbalanced dataset** situation, which is very common in machinery fault diagnosis.[^4_7][^4_8][^4_6]

For this project:

- **Do not throw away images from the larger classes at the start.**
Undersampling all other classes down to the small Healthy class would discard useful information and usually hurts performance unless the imbalance is extreme.[^4_9][^4_10][^4_11]
- Instead, use **class weights** during training, so that mistakes on Healthy images are penalized more than mistakes on more common classes.[^4_10][^4_11][^4_4]
- If later you find you still miss many Healthy samples, you can optionally add **light oversampling** of Healthy (e.g. repeat those images more often in the training dataset or apply extra augmentation).[^4_12][^4_8][^4_7]

The CWT‑RGB‑CNN literature on rotating machinery and bearings follows similar ideas: keep all time–frequency images and handle imbalance with oversampling or cost‑sensitive loss rather than discarding majority data.[^4_8][^4_3][^4_7]

***

### 2. How the directory should look

Your `tree.txt` shows a root folder with subfolders such as:[^4_6]

- `./10 um/`
- `./45 um/`
- `./75 um/`
- `./100 um/`
- `./Healthy/`

Each contains many `*_cwt.png` images (or `.jpg`).[^4_1][^4_6]

For Keras’s `image_dataset_from_directory`, it is convenient to have a **single root** for all images, for example:

```text
cwt_images/
    10 um/
        10 um_merged_cleaned_20251112_190203_1_cwt.png
        ...
    45 um/
        ...
    75 um/
        ...
    100 um/
        ...
    Healthy/
        ...
```

This is already what you have, just ensure you know the absolute path, say `"/home/you/cwt_images"` on your machine.[^4_6]

***

### 3. Training/validation/test split

A standard, simple split that works well:

- 70% of images for **training**
- 15% for **validation** (to tune hyperparameters and early stopping)
- 15% for **test** (to report final performance)

Keras can create **train** and **validation** datasets automatically using `validation_split`, and then you can carve a part of the validation set as test if you wish, or create a separate test set manually.  In the pump and bearing papers, similar ratios are used, carefully ensuring that samples from the same original signal don’t leak between splits.[^4_13][^4_3][^4_4][^4_10][^4_1]

***

### 4. CNN model choice (summary)

Your CWT images already look structured and visually separable across classes: the intensity and pattern of the bands changes systematically as wear increases.  A **small CNN with batch normalization** is sufficient and matches the architectures used in hydraulic pump CWT/SWT studies.[^4_14][^4_15][^4_16][^4_13][^4_1]

We will build:

- 3 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
- GlobalAveragePooling2D
- Dense(64, ReLU) → Dense(5, Softmax)

This is a “light” normalized CNN analogous to the adaptive NCNN in your attached pump papers, but easier to code.[^4_14][^4_13][^4_1]

***

### 5. Complete TensorFlow/Keras code

Below is a full example you can run as a single script (for example in a notebook).[^4_4][^4_13][^4_10][^4_1]

```python
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1. Configuration
# -----------------------------

# Path to the root directory that contains the 5 class folders
DATA_DIR = "/path/to/cwt_images"   # <-- change this to your folder

IMAGE_SIZE = (256, 256)           # your images are already 256x256
BATCH_SIZE = 32
SEED = 123

NUM_CLASSES = 5   # Healthy, 10 um, 45 um, 75 um, 100 um

# -----------------------------
# 2. Load train and validation datasets
# -----------------------------
# Here we let Keras do a 80/20 split; later we will further split
# the validation set into validation + test if needed.

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",           # integer labels 0..4
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,       # 80% train, 20% val
    subset="training",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

# Optional: split val_ds into val and test (50/50)
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds  = val_ds.skip(val_batches // 2)

class_names = train_ds.class_names
print("Class names:", class_names)

# -----------------------------
# 3. Performance optimizations
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(1000, seed=SEED)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds   = configure_for_performance(val_ds)
test_ds  = configure_for_performance(test_ds)

# -----------------------------
# 4. Compute class weights (handle imbalance)
# -----------------------------
# Iterate once through the original (un-shuffled) unbatched dataset to count labels.

# Recreate an unbatched dataset for counting
count_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=1,
    image_size=IMAGE_SIZE,
    shuffle=False
)

label_counts = dict.fromkeys(range(NUM_CLASSES), 0)

for _, labels in count_ds:
    label = int(labels.numpy()[^4_0])
    label_counts[label] += 1

print("Label counts:", label_counts)

total_samples = sum(label_counts.values())
# Inverse-frequency class weights:
class_weights = {
    cls: (total_samples / (NUM_CLASSES * count))
    for cls, count in label_counts.items()
}

print("Class weights:", class_weights)

# -----------------------------
# 5. Data augmentation (gentle)
# -----------------------------
# For CWT images, avoid flips that invert frequency axis. You can still
# add small random contrast / brightness or tiny zooms along time.

data_augmentation = keras.Sequential(
    [
        layers.RandomBrightness(factor=0.1),
        layers.RandomContrast(factor=0.1),
        # layers.RandomZoom(0.05)  # optional, keep small
    ]
)

# -----------------------------
# 6. Build the CNN model
# -----------------------------

def make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)   # scale pixels to [0,1]

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cwt_cnn")
    return model

model = make_model()
model.summary()

# -----------------------------
# 7. Compile the model
# -----------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------
# 8. Train with early stopping
# -----------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_cwt_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

EPOCHS = 40

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,  # important for imbalance
    callbacks=callbacks,
)

# -----------------------------
# 9. Evaluate on the test set
# -----------------------------

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# Optional: confusion matrix
import sklearn.metrics as skm

y_true = []
y_pred = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = skm.confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

print("Classification report:\n",
      skm.classification_report(y_true, y_pred, target_names=class_names))
```

This script:

- Reads your CWT scalograms from the five class folders.[^4_1][^4_6]
- Splits them into train/validation/test datasets.[^4_10][^4_4]
- Computes **class weights** from the raw image counts to reduce the impact of class imbalance.[^4_11][^4_4][^4_10]
- Trains a light CNN with BatchNorm and small augmentation—very similar in spirit to the NCNN/LeNet‑style models used in your hydraulic pump papers.[^4_13][^4_14][^4_1]
- Evaluates test accuracy and prints a confusion matrix and per‑class metrics so you can see how well each wear level is recognized.[^4_3][^4_13][^4_1]

***

### 6. How to interpret and iterate

Once you run this:

- If **overall accuracy is good** (say >95%) but Healthy is still misclassified often (look at the confusion matrix), consider:
    - Slightly **increasing the Healthy class weight** manually.[^4_4][^4_10]
    - Adding a bit more **augmentation** specifically for Healthy (e.g. small contrast changes) to increase its effective diversity.[^4_7][^4_8]
- If training accuracy is very high but validation accuracy is low, reduce model size (fewer filters) or increase dropout to reduce overfitting.[^4_17][^4_18]
- If everything trains smoothly, you can later experiment with:
    - A slightly deeper CNN (e.g. 4 conv blocks),
    - Attention modules (CBAM) as in recent bearing work,
    - Or hyperparameter optimization (KerasTuner) to mirror the adaptive frameworks in the pump papers.[^4_3][^4_14][^4_13]

***

### 7. Direct answers to your specific questions

- **“Healthy has the least images; should I reduce other classes to match it?”**
No; keep all images and use **class weights** (and optionally oversampling/augmentation) to address imbalance, instead of throwing away useful data from other classes.[^4_8][^4_7][^4_10]
- **“Can we use TensorFlow—is it well suited?”**
Yes, TensorFlow/Keras is very well suited; it is widely used for CNN‑based fault diagnosis on CWT or spectrogram images and provides exactly the tools you need for this pipeline.[^4_2][^4_17][^4_3]

If you tell which OS and Python environment you are using, the next step can be to help you adapt paths and commands so this script runs directly on your machine.[^4_10][^4_4]
<span style="display:none">[^4_19][^4_20][^4_21][^4_22][^4_23][^4_24][^4_25][^4_26][^4_27][^4_28][^4_29][^4_30][^4_31][^4_32]</span>

<div align="center">⁂</div>

[^4_1]: 75-um_merged_cleaned_20251112_190552_2_cwt.jpg

[^4_2]: https://ieeexplore.ieee.org/iel8/6287639/10820123/10935311.pdf

[^4_3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10974164/

[^4_4]: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

[^4_5]: 10-um_merged_cleaned_20251112_190203_2.csv

[^4_6]: tree.txt

[^4_7]: https://www.mdpi.com/1424-8220/22/22/8749/pdf?version=1668388874

[^4_8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9692439/

[^4_9]: https://wandb.ai/authors/class-imbalance/reports/Simple-Ways-to-Tackle-Class-Imbalance--VmlldzoxODA3NTk

[^4_10]: https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets

[^4_11]: https://stackoverflow.com/questions/55657807/class-weights-vs-under-oversampling

[^4_12]: https://www.mdpi.com/2075-4442/9/10/105/pdf

[^4_13]: 45-um_merged_cleaned_20251112_190434_2_cwt.jpg

[^4_14]: Healthy_merged_cleaned_20251112_185902_2_cwt.jpg

[^4_15]: 100-um_merged_cleaned_20251112_190822_2_cwt.jpg

[^4_16]: Healthy_merged_cleaned_20251112_185902_1_cwt.jpg

[^4_17]: https://www.extrica.com/article/23391/pdf

[^4_18]: https://arxiv.org/pdf/2502.08689.pdf

[^4_19]: 75-um_merged_cleaned_20251112_190552_1_cwt.jpg

[^4_20]: 100-um_merged_cleaned_20251112_190822_1_cwt.jpg

[^4_21]: 45-um_merged_cleaned_20251112_190434_1_cwt.jpg

[^4_22]: 10-um_merged_cleaned_20251112_190203_1_cwt.jpg

[^4_23]: 10-um_merged_cleaned_20251112_190203_2_cwt.jpg

[^4_24]: https://www.extrica.com/article/23612/pdf

[^4_25]: https://www.mdpi.com/1424-8220/18/4/1096/pdf

[^4_26]: https://www.mdpi.com/2076-3417/14/11/4927/pdf?version=1717658647

[^4_27]: https://www.mdpi.com/1424-8220/24/6/1831/pdf?version=1710313474

[^4_28]: https://www.sciencedirect.com/science/article/pii/S0925231225002607

[^4_29]: https://pubs.aip.org/aip/adv/article/14/12/125321/3327648/Classification-performance-improvement-in

[^4_30]: https://www.nature.com/articles/s41598-025-02439-4

[^4_31]: https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-data-for-classification/

[^4_32]: https://community.deeplearning.ai/t/class-imbalance-in-object-detection-using-yolo/352498


---

# Where shall I run this code, and how? Can I run it on my M1 pro macbook pro (base model)? shall I use conda?

Yes, you can run this entire TensorFlow/Keras pipeline comfortably on an M1 Pro MacBook Pro, even the base model, using the Apple‑optimized TensorFlow build and either conda or plain `venv`.

Below is a minimal, practical way to set things up.

***

### 1. Where and how to run the code

- Run it **locally on your M1 Pro** using a Python environment (no need for cloud now).
- Use either:
    - **Conda (miniforge or anaconda)** with the Apple Silicon stack, or
    - **Python’s built‑in `venv`** plus `pip`.

For a beginner, a clean conda environment is convenient but not strictly required.

***

### 2. Recommended setup on M1 Pro (with conda)

1. **Install a conda distribution for Apple Silicon**
    - Prefer **miniforge** or **miniconda** built for arm64 (you may already have one).
2. **Create and activate an environment**

```bash
conda create -n tf_m1 python=3.10
conda activate tf_m1
```

3. **Install the Apple TensorFlow stack**

Apple maintains a special build that uses the M1 GPU/Neural Engine via Metal; it runs much faster than CPU‑only.

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow-macos tensorflow-metal
```

4. **Install extra packages used in the script**

```bash
python -m pip install scikit-learn matplotlib
```

5. **Save the training script**
    - Create a file, for example `train_cwt_cnn.py`.
    - Copy the full TensorFlow script from the previous answer into this file.
    - Edit the line

```python
DATA_DIR = "/path/to/cwt_images"
```

so it points to the folder that contains your five subfolders (`Healthy`, `10 um`, `45 um`, `75 um`, `100 um`).
6. **Run the script from Terminal**

```bash
conda activate tf_m1
python train_cwt_cnn.py
```

The terminal should print:
    - Class names
    - Label counts and class weights
    - Model summary
    - Then epoch‑by‑epoch training and validation accuracy, followed by test accuracy and the confusion matrix.

***

### 3. If you prefer not to use conda

You can do the same with plain `venv`:

```bash
python3 -m venv tf_m1_env
source tf_m1_env/bin/activate

python -m pip install --upgrade pip
python -m pip install tensorflow-macos tensorflow-metal scikit-learn matplotlib
```

Then run:

```bash
python train_cwt_cnn.py
```

The behavior will be the same as with conda; conda just makes managing multiple environments easier.

***

### 4. Practical notes specific to your Mac

- The **M1 Pro GPU is supported** by `tensorflow-macos` + `tensorflow-metal`, so training on your 256×256 CWT images should be reasonably fast.
- Keep `BATCH_SIZE = 32` or even 16 if you hit out‑of‑memory errors, then increase gradually.
- Close heavy apps (Chrome with many tabs, etc.) during training to keep more memory free.

***

If you tell what you already have installed (Python version, whether conda is there), the next step can be a set of exact commands tailored to your current setup so you can copy‑paste and run with minimal friction.

