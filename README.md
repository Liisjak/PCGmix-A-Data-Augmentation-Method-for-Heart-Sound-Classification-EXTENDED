# PCGmix: A Data-Augmentation Method for Heart-Sound Classification (EXTENDED)

This repository extends the work presented in the original paper:

**[PCGmix: A Data-Augmentation Method for Heart-Sound Classification](https://ieeexplore.ieee.org/document/10675426)**  
*Published in IEEE JBHI, 2024*

The original PCGmix introduced a novel **data augmentation method** tailored to **phonocardiogram (PCG)** signals for **heart-sound classification**. This extended version adds advanced interpretability and robustness analyses, enabling deeper insights into the method's generalization performance and behavior.

## Repository Structure

This repository contains code for augmenting, training, and analyzing heart-sound classification models using the PCGmix method.

This repository uses `Python 3.8.16` with dependencies listed in `requirements.txt`.

Below is a brief explanation of each key file:

---

### Data Preparation

- **`databuilder.ipynb`**: Preprocessing pipeline to convert raw PCG recordings and annotations into structured datasets suitable for training.

- **`dataloader_physionet.py`** / **`dataloader_umc.py`**: DataLoader implementations for **time-series** data from the PhysioNet and UMC datasets, respectively.

- **`dataloader_physionet2d.py`** / **`dataloader_umc2d.py`**: DataLoaders for loading **spectrogram** versions of the PhysioNet and UMC datasets.

### Modeling

- **`models.py`**: Deep learning models for 1D time-series classification, including variants of CNNs and ResNet architectures.

- **`models2d.py`**: Deep learning models for 2D spectrogram classification (e.g., 2D ResNet9).

### Training

- **`train_model.py`**: Core training script supporting both 1D and 2D models. Integrates augmentation, model selection, and evaluation.

### Augmentation

- **`augmentations.py`**: Implements data augmentation methods for time-series heart sound signals, including the core **PCGmix** strategy. Our methods are:
  - PCGmix: `durratiomixup`
  - PCGmix+: `durmixmagwarp(0.2,4)`

- **`augmentations2d.py`**: Contains analogous augmentations for 2D spectrogram representations of heart sounds.

### Utilities

- **`utils.py`**: General utility functions used across scripts and notebooks (open files, create dirs, read results, etc.).

- **`plotters.py`**: Tools to visualize model training progress (loss, accuracy) and other visualizations.

- **`latent_space.py`**: Extracts the learned latent spaces of models, useful for studying out-of-manifold intrusion and generalization.

- **`saliency.py`**: Implements saliency information computation for saliency-guided augmentation.

- **`read_experiments.py`**: Parses and organizes experiment logs and results for easy comparison and analysis/plotting.

### Analysis

- **`experiments_timeseries.ipynb`**: Full experiments for training and evaluating time-series models with/without augmentation.

- **`experiments_spectrograms.ipynb`**: Similar to above, but applied to spectrogram-based models.

- **`results_final_full.ipynb`**: Aggregated results, visualizations, and analysis for the final submission or report.

### Classical machine learning

- **`classical.py`**, **`classical.ipynb`**: Utilities and experiments for evaluating classical ML models (e.g., Random Forests, SVMs) on both original and augmented datasets.

---

## Key extensions beyond the original PCGmix paper

In addition to the core PCGmix method, this extended repository provides:

- **Spectorgram-based analysis**: Methods are also evaluated in spectrogram domain
- **Out-of-manifold intrusion analysis**: Evaluation of how generated data interacts with the true data manifold, including its effects on decision boundaries and generalization.
- **Saliency-guided augmentation optimization**: Techniques to utilize saliency data for more effective augmentation and training strategies.
- **Parameter analysis**: different mapping fuctions, different `α` values for beta distribution and more.

## Dataset

### PhysioNET

- Raw data: [PhysioNet Challenge 2016](https://archive.physionet.org/pn3/challenge/2016/)
- Segmentation files (Springer + hand-corrected): [Link](https://physionet.org/content/challenge-2016/1.0.0/#files-panel)

#### Test Data

We used the recordings marked as the **"validation"** set as our test data. A few of the recordings from the "validation" set that do not include the corresponding segmentation files and are thus excluded in our study: e00001, e00032, e00039, e00044.

#### Train Data

The train set in experiments is a subset of the “train set” used in the Challenge. 

Those recordings include: a0059, a0101, a0103, a0104, a0110, a0115, a0120, a0122, a0123, a0126, a0130, a0132, a0133, a0134, a0140, a0141, a0142, a0143, a0148, a0149, a0152, a0153, a0154, a0155, a0158, a0160, a0161, a0165, a0166, a0169, a0170, a0171, a0172, a0178, a0179, a0180, a0181, a0183, a0184, a0185, a0187, a0188, a0189, a0193, a0195, a0196, a0197, a0199, a0204, a0208, a0210, a0211, a0212, a0213, a0214, a0221, a0224, a0227, a0228, a0229, a0231, a0235, a0236, a0238, a0240, a0241, a0242, a0243, a0245, a0246, a0248, a0250, a0252, a0253, a0254, a0257, a0261, a0264, a0266, a0267, a0268, a0269, a0270, a0271, a0274, a0276, a0278, a0283, a0285, a0287, a0288, a0289, a0290, a0291, a0293, a0294, a0297, a0298, a0299, a0301, a0302, a0304, a0306, a0309, a0310, a0311, a0312, a0313, a0320, a0323, a0325, a0326, a0328, a0329, a0331, a0332, a0334, a0335, a0336, a0337, a0339, a0340, a0342, a0346, a0347, a0348, a0351, a0352, a0353, a0355, a0358, a0359, a0361, a0366, a0368, a0371, a0374, a0381, a0384, a0385, a0386, a0389, a0391, a0393, a0396, a0401, a0404, a0405, a0406, a0407, a0408, a0409, b0110, b0112, b0129, b0132, b0134, b0142, b0143, b0156, b0157, b0161, b0166, b0177, b0189, b0192, b0202, b0246, b0250, b0251, b0257, b0261, b0262, b0263, b0265, b0267, b0268, b0269, b0271, b0273, b0279, b0280, b0282, b0292, b0295, b0306, b0307, b0318, b0319, b0327, b0328, b0334, b0335, b0339, b0341, b0344, b0347, b0354, b0369, b0373, b0383, b0385, b0390, b0392, b0395, b0397, b0406, b0408, b0422, b0428, b0434, b0435, b0436, b0437, b0446, b0449, b0450, b0454, b0467, b0468, b0471, b0474, b0477, b0478, b0484, b0490, c0010, c0011, c0015, c0016, c0019, c0022, c0023, c0030, d0010, d0012, d0014, d0015, d0016, d0017, d0018, d0019, d0020, d0021, d0022, d0023, d0024, d0025, d0027, d0028, d0029, d0030, d0031, d0032, d0033, d0034, d0035, d0036, d0037, d0038, d0039, d0040, d0041, d0042, d0043, d0045, d0046, d0047, d0048, d0049, d0050, d0051, d0052, d0053, d0054, d0055, e00074, e00254, e00277, e00309, e00347, e00354, e00365, e00387, e00419, e00425, e00436, e00453, e00480, e00493, e00509, e00512,  e00531, e00552, e00559, e00566, e00568, e00569, e00580, e00600, e00613, e00645, e00652, e00658, e00686, e00693, e00696, e00699, e00703, e00710, e00725, e00726, e00739, e00744, e00765, e00766,e00792, e00799, e00808, e00811, e00818, e00824, e00841, e00847, e00871, e00872, e00880, e00897,e00908, e00914, e00917, e00925, e00926, e00929, e00936, e00967, e00974, e00980, e00998, e01013, e01016, e01055, e01062, e01072, e01073, e01075, e01084, e01097, e01105, e01115, e01148, e01160,
e01177, e01198, e01205, e01215, e01225, e01226, e01245, e01246, e01247, e01252, e01253, e01256, e01263, e01272, e01276, e01283, e01284, e01289, e01291, e01295, e01299, e01308, e01324, e01336, e01341, e01351, e01358, e01367, e01369, e01374, e01375, e01376, e01382, e01383, e01392, e01399, e01401, e01416, e01421, e01433, e01457, e01461, e01474, e01479, e01487, e01489, e01493, e01494, e01511, e01525, e01531, e01537, e01551, e01559, e01572, e01581, e01601, e01602, e01605, e01618, e01623, e01625, e01638, e01651, e01661, e01663, e01665, e01666, e01668, e01686, e01688, e01689, e01697, e01698, e01711, e01719, e01728, e01734, e01748, e01753, e01756, e01766, e01767, e01787, e01791, e01800, e01808, e01812, e01824, e01825, e01832, e01846, e01848, e01851, e01855, e01856,
e01857, e01867, e01869, e01873, e01881, e01917, e01922, e01924, e01929, e01938, e01953, e01959, e01962, e01967, e01971, e01978, e02001, e02010, e02015, e02017, e02030, e02032, e02044, e02045, e02055, e02058, e02059, e02065, e02074, e02085, e02088, e02092, e02097, e02101, e02102, e02108, e02111, e02117, e02121, e02126, e02133, e02135, e02137, e02140, f0003, f0007, f0008, f0011, f0012, f0015, f0016, f0017, f0018, f0020, f0022, f0023, f0027, f0029, f0031, f0035, f0036, f0041, f0042, f0045, f0047, f0048, f0050, f0052, f0054, f0056, f0060, f0063, f0064, f0065, f0066, f0067, f0068, f0069, f0070, f0071, f0072, f0075, f0076, f0078, f0079, f0080, f0081, f0082, f0083, f0085, f0090, f0091, f0092, f0093, f0096, f0097, f0098, f0099, f0100, f0101, f0103, f0106, f0109, f0112, f0113, and f0114.
  
The list of recordings is also given in the file `PhysioNet_seed(data)=1100001_nfrac=1.0_valid=False.txt`

### UMC

A private dataset collected for:

**[Identification of decompensation episodes in chronic heart failure patients based solely on heart sounds](https://www.frontiersin.org/articles/10.3389/fcvm.2022.1009821/full)**     
*Published in Front. Cardiovasc. Med., 2022*

**Raw data can be made available upon request.**

## Models

We used two main architectures for our final results and analysis

### Time-series domain

- **1D-CNN**: Based on the model from Potes et al. [CinC 2016](https://doi.org/10.22489/CinC.2016.182-399)
- **1D-ResNet9**: A 1D variant of [ResNet](https://arxiv.org/abs/1512.03385), where all 2D layers are converted to their 1D counterparts.

Additional models are implemented in the scripts, most of which are from the [tsai](https://github.com/timeseriesAI/tsai/tree/main/tsai/models) library.

### Spectrogram domain

- **2D-ResNet9**: 2D (original) [ResNet](https://arxiv.org/abs/1512.03385)

## Training Parameters

Training was conducted with consistent hyperparameters across all experiments (except where the parameter was subject to analysis:

- **Optimizer Scheduler**: One-Cycle-LR with `lr_max = 0.01`
- **Batch Size**: `64`
- **Epochs**: `50`

## How to cite

If you use PCGmix or this extended version in your research, please cite the original work:
```bibtex
@article{10675426,
  author={Susič, David and Gradišek, Anton and Gams, Matjaž},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={PCGmix: A Data-Augmentation Method for Heart-Sound Classification}, 
  year={2024},
  volume={28},
  number={11},
  pages={6874-6885},
  keywords={Heart;Feature extraction;Phonocardiography;Spectrogram;Data models;Heart beat;Data augmentation;Data augmentation;phonocardiogram;heart sounds;abnormal heart-sound detection;deep learning;neural networks;machine learning},
  doi={10.1109/JBHI.2024.3458430}}
```
