
# 🌟 **Retina SEM — Unsupervised Retinal Vessel Segmentation via Structural Entropy Minimization**
### *Team AceHunters — IIT Bhilai*

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Machine%20Learning-ICDM%20SLED-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Category-Medical%20Imaging-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge" />
</p>

---

## 📌 **1. Project Overview**

This repository implements an **unsupervised retinal vessel segmentation** pipeline using  
**Structural Entropy Minimization (SEM)**, originally proposed for skin lesion segmentation  
and adapted here for **retinal fundus images**.

Traditional vessel segmentation requires **pixel-level annotations**, which are costly.  
Our method requires **zero labels** and uses:

- Multi-scale **SLIC superpixels**
- **Graph construction**
- **Structural Entropy Minimization**
- **Multi-scale fusion**

The project is implemented on the **FIVES dataset**.

---

## 🔥 **2. High-Level Architecture**

```                   
       ┌────────────────┐
       │  Dataset(FIVES)│
       └───────┬────────┘
               │ Phase 0
               ▼
     ┌────────────────────┐
     │ Preprocessing(CLAHE,
     │ FOV Masking)       │
     └─────────┬──────────┘
               │ Phase 1
               ▼
     ┌────────────────────┐
     │ Superpixel Graphs  │
     │ (SLIC + Features)  │
     └─────────┬──────────┘
               │ Phase 2
               ▼
     ┌────────────────────┐
     │ Structural Entropy │
     │ Minimization       │
     └─────────┬──────────┘
               │ Phase 3
               ▼
     ┌────────────────────┐
     │ Multi-Scale Fusion │
     │   Final Mask       │
     └────────────────────┘
```



---

## 📂 **3. Folder Structure**

```
Retina_SEM/
│
├── dataset/                  
│   ├── train/Original
│   ├── train/Ground truth
│   ├── test/Original
│   └── test/Ground truth
│
├── preprocessed/
│   ├── train/images
│   ├── train/fov_masks
│   ├── val/images
│   ├── val/fov_masks
│   └── test/images
│   └── test/fov_masks
│
├── superPixel_graph/
│   ├── scale_K1500_C10
│   ├── scale_K3500_C6
│   ├── scale_K4500_C5
│
├── superPixel_graph_sem/
│   ├── scale_K1500_C10
│   ├── scale_K3500_C6
│   ├── scale_K4500_C5
│
├── outputs/
│   └── final_vessel_masks/
│
├── src/
│   ├── phase0_index.py
│   ├── phase1_preprocess.py
│   ├── step2_build_superpixel_graphs.py
│   ├── step3_structural_entropy.py
│   ├── run_multiscale_batch.py
│   └── utils/
│
└── NoteBooks/
    ├── 00_visualize_fov.ipynb
    ├── 01_superpixel_graph_visualization.ipynb
    └── 02_entropy_heatmaps.ipynb
```

---

## 💻 **4. Environment Setup**

### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

# 🚀 **5. Pipeline Execution (All Steps)**

---

## 🔹 **PHASE 0 — Dataset Indexing**
Organizes dataset into train/val/test.

```bash
python src/phase0_index.py \
    --data-root dataset \
    --out-root dataset
```

---

## 🔹 **PHASE 1 — Preprocessing (CLAHE + FOV Mask)**

```bash
python src/phase1_preprocess.py \
    --data-root dataset \
    --out-root preprocessed
```

Outputs:

```
preprocessed/train/images/
preprocessed/train/fov_masks/
```

---

## 🔹 **PHASE 2 — Superpixel Graph Construction**

```bash
python src/step2_build_superpixel_graphs.py \
    --data-root preprocessed \
    --out-root superPixel_graph \
    --k 3500 \
    --compactness 6
```

---

## 🔹 **PHASE 3 — Structural Entropy Minimization**

```bash
python src/step3_structural_entropy.py \
    --scale_name scale_K3500_C6 \
    --graph_root superPixel_graph/scale_K3500_C6 \
    --out_root superPixel_graph_sem/scale_K3500_C6 \
    --splits train val test \
    --num_iters 50 \
    --lr 0.001
```

---

## 🔹 **PHASE 4 — Multi-Scale Fusion**

Generates the final binary vessel masks.

```bash
python src/run_multiscale_batch.py \
    --sem-root superPixel_graph_sem \
    --out-dir outputs/final_vessel_masks \
    --thr 0.78 \
    --intensity-thr 53 \
    --min-area 80
```

---

# 📊 **6. Visualization with Jupyter Notebooks**

Launch:

```bash
jupyter notebook
```

### Notebooks:
| Notebook | Purpose |
|---------|---------|
| **00_visualize_fov.ipynb** | CLAHE, FOV masks, preprocessing |
| **01_superpixel_graph_visualization.ipynb** | SLIC superpixels, graph edges |
| **02_entropy_heatmaps.ipynb** | Entropy maps before/after optimization |

---

# 🧩 **7. Multi-Scale Strategy**

To capture both large and fine vessels:

| Scale | K value | Captures |
|-------|--------|----------|
| **K1500** | Coarse | Large trunk vessels |
| **K3500** | Medium | Balanced |
| **K4500** | Fine | Capillaries |

Fusion increases **clDice** and preserves vessel connectivity.

---

# 🧪 **8. Metrics Used**

| Metric | Purpose |
|--------|---------|
| **Dice** | Pixel overlap |
| **clDice** | Vessel topology + connectivity |
| **PR-AUC** | Threshold-independent |

Retina SEM consistently outperformed:
- Frangi filter  
- Green-channel thresholding  

Especially in **clDice**, proving better microvascular continuity.

---

# 📝 **9. Contributors**

### **Team AceHunters — IIT Bhilai**
- **Om Raj Singh** — M25DS007  
- **Rohan Sinha** — M25DS008  
- **Sarvesh Badoni** — M25DS011  
- **Vedant Tawri** — M25DS016

---

# 📚 **10. Citation**

```
@inproceedings{sled2023,
  title={Unsupervised Skin Lesion Segmentation via Structural Entropy Minimization on Multi-Scale Superpixel Graphs},
  author={...},
  booktitle={ICDM},
  year={2023}
}
```

---

# 📜 **11. License**
MIT License (add a LICENSE file if required)

---

# 🎉 **12. Final Notes**

For any issues or suggestions:  
Open an Issue or Contact Team AceHunters.

---

