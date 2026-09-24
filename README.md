# 🎓 Student Academic Performance Prediction: Pass or Fail?
> **Data Mining Project:** การทำนายผลการเรียนของนักเรียน (สอบผ่าน vs สอบไม่ผ่าน) โดยเปรียบเทียบระหว่างโมเดล **Decision Tree (Rule-based)** และ **Naive Bayes (Probabilistic)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20UCI-green.svg)](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 1. บทนำและเป้าหมายของโครงงาน (Project Overview)
* **ปัญหาทางธุรกิจ/การศึกษา:** โรงเรียนต้องการระบบคัดกรองนักเรียนที่มีความเสี่ยงจะสอบตกตั้งแต่ต้นภาคการศึกษา (**Early Warning System**) เพื่อให้ครูที่ปรึกษาสามารถยื่นมือเข้าไปช่วยเหลือและวางแผนติวเสริมได้ทันท่วงที
* **แหล่งข้อมูล (Dataset):** [Student Alcohol Consumption / Student Performance (Kaggle/UCI)](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) ข้อมูลนักเรียนวิชาภาษาโปรตุเกส (`student-por.csv`) จำนวน **649 แถว และ 33 ตัวแปร**
* **เป้าหมาย (Target Variable):** คะแนนสอบปลายภาค `G3` โดยแบ่งเป็น:
  * `Pass (1)`: คะแนน $\ge 10$ (คิดเป็น 84.6% ของนักเรียนทั้งหมด)
  * `Fail (0)`: คะแนน $< 10$ (คิดเป็น 15.4% ของนักเรียนทั้งหมด)
* **การป้องกัน Data Leakage:** ตัดคะแนนสอบย่อย $G1$ และ $G2$ ออกจากตัวแปรทำนาย เพื่อบังคับให้โมเดลทำนายจากพฤติกรรม สภาพแวดล้อมครอบครัว และประวัติการขาดเรียนตั้งแต่เริ่มเปิดเทอม

---

## 👥 2. สมาชิกและการแบ่งงานบน GitHub Branches
การทำงานถูกแยกออกเป็น Branch เพื่อการทำงานร่วมกันแบบมืออาชีพ:

| สมาชิก | Branch | ความรับผิดชอบและสมุดบันทึก (Notebook) |
| :--- | :--- | :--- |
| **ส่วนร่วมกัน** | `main` | วางโครงสร้างโปรเจกต์, ทำ EDA, ทำความสะอาดข้อมูล, และวิเคราะห์เปรียบเทียบ (`01_eda_and_data_cleaning.ipynb`, `04_model_comparison.ipynb`) |
| **คนที่ 1** | `feature/decision-tree` | พัฒนาโมเดล Decision Tree, ทำ Pre-pruning (`max_depth`), Post-pruning (`ccp_alpha`), พล็อตต้นไม้ และวิเคราะห์ Feature Importance (`02_decision_tree_model.ipynb`) |
| **คนที่ 2** | `feature/naive-bayes` | ให้เหตุผลและเลือกใช้ `GaussianNB`, วิเคราะห์ความสำคัญของฟีเจอร์ด้วย Permutation Importance และวิเคราะห์ค่าเฉลี่ย ($\theta$) (`03_naive_bayes_model.ipynb`) |

---

## 📂 3. โครงสร้างโฟลเดอร์ของโปรเจกต์ (Repository Structure)
```text
student-grade-prediction/
│
├── .gitignore                          # รายการไฟล์ที่ไม่ต้องการดันขึ้น Git
├── requirements.txt                    # รายการ Dependencies ทั้งหมด
├── README.md                           # เอกสารอธิบายโปรเจกต์ฉบับสมบูรณ์
│
├── data/
│   ├── raw/                            # ข้อมูลดิบ student-por.csv จาก Kaggle
│   └── processed/                      # ข้อมูล train.csv และ test.csv หลังทำ Preprocessing
│
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb  # (ทำร่วมกัน) EDA, Data Cleaning, Outliers (IQR), One-Hot
│   ├── 02_decision_tree_model.ipynb   # (คนที่ 1) Decision Tree, Pruning, plot_tree, Gini vs Entropy
│   ├── 03_naive_bayes_model.ipynb     # (คนที่ 2) GaussianNB, Permutation Importance
│   └── 04_model_comparison.ipynb      # (ทำร่วมกัน) ประชันผลลัพธ์ Head-to-Head และสรุปเชิงนโยบาย
│
└── images/                             # กราฟความละเอียดสูงสำหรับรายงานและสไลด์
    ├── outliers_boxplot.png            # การตรวจจับ Outlier ด้วย IQR
    ├── correlation_heatmap.png         # Heatmap แสดงสหสัมพันธ์
    ├── tree_structure.png              # แผนผังต้นไม้การตัดสินใจหลังทำ Pruning
    ├── feature_importance.png          # กราฟแท่งแสดงค่าน้ำหนักตัวแปร
    ├── confusion_matrices.png          # Confusion Matrix เปรียบเทียบสองโมเดล
    └── roc_curves.png                  # เส้นกราฟ ROC Curves เปรียบเทียบ
```

---

## 📊 4. สรุปผลการทดลองจริง (Head-to-Head Experimental Results)

การทดสอบทำบนชุดทดสอบมาตรฐาน (**Test Set จำนวน 130 คน**: สอบผ่านจริง 110 คน, สอบตกจริง 20 คน):

| ตัวชี้วัด (Metric) | Decision Tree (Unpruned) | Decision Tree (Post-Pruned) | Gaussian Naive Bayes |
| :--- | :---: | :---: | :---: |
| **Training Accuracy** | 1.0000 (Overfit 100%) | 0.8459 | 0.7765 |
| **Testing Accuracy** | 0.7769 (77.69%) | **0.8000 (80.00%)** | 0.6385 (63.85%) |
| **Precision (Pass)** | 0.8584 | 0.8684 | **0.8795** |
| **Recall (Pass)** | 0.8818 | **0.9000** | 0.6636 |
| **Recall (Fail) ⚠️** | 0.2000 (จับได้ 4/20) | 0.2500 (จับได้ 5/20) | **0.5000 (จับได้ 10/20)** |
| **F1-Score (Macro)** | 0.5409 | **0.7294** | 0.6285 |
| **ROC-AUC Score** | 0.5409 | **0.6818** | 0.5023 |

### 🖼️ ภาพผลลัพธ์สำคัญ (Key Visualizations)
| แผนผังต้นไม้การตัดสินใจ (Decision Tree) | การเปรียบเทียบ Confusion Matrices |
| :---: | :---: |
| ![Decision Tree](images/tree_structure.png) | ![Confusion Matrices](images/confusion_matrices.png) |

| การเปรียบเทียบเส้น ROC Curves | ค่าน้ำหนักความสำคัญของตัวแปร (Feature Importance) |
| :---: | :---: |
| ![ROC Curves](images/roc_curves.png) | ![Feature Importance](images/feature_importance.png) |

---

## 💡 5. บทสรุปและการค้นพบองค์ความรู้ (Key Insights & Discussion)
1. **การแก้ปัญหา Overfitting ใน Decision Tree:** ต้นไม้ที่ปล่อยให้เติบโตตามธรรมชาติเกิด Overfitting 100% แต่การใช้เทคนิค **Cost-Complexity Pruning (`ccp_alpha = 0.00681`)** ร่วมกับ 5-Fold Cross-Validation ช่วยตัดกิ่งที่ไม่จำเป็นออก ทำให้ Test Accuracy เพิ่มขึ้นเป็น **80.00%**
2. **ตัวแปรที่มีอิทธิพลสูงสุด:**
   * **`failures` (ประวัติการสอบตก):** มีน้ำหนักสูงถึง **58.1%** ชี้ให้เห็นว่าผลการเรียนในอดีตคือตัวทำนายอนาคตที่แม่นยำที่สุด
   * **`higher_yes` (ความต้องการเรียนต่อ):** มีน้ำหนัก **19.1%** สะท้อนถึงแรงจูงใจและความมุ่งมั่นของนักเรียน
3. **Accuracy vs Recall Trade-off (ประเด็นชี้เป็นชี้ตาย):**
   * **Decision Tree** ชนะเลิศด้านภาพรวม (Accuracy 80%) และสามารถอธิบายกฎ If-Else ได้อย่างโปร่งใส (**White-Box**)
   * **Naive Bayes** แม้จะมีความแม่นยำรวมต่ำกว่า (63.85%) จากข้อจำกัดเรื่องสมมติฐานความอิสระ แต่มี **Recall ของกลุ่มตก (Fail) สูงถึง 50.00%** (จับเด็กตกได้ 10 คน เทียบกับ Decision Tree ที่จับได้ 5 คน)
4. **ข้อเสนอแนะเชิงนโยบายสำหรับโรงเรียน:**
   * ควรใช้ Decision Tree ควบคู่กับการเปิดออปชัน `class_weight='balanced'` เพื่อให้โมเดลจับเด็กตกได้เพิ่มขึ้นจาก 25% เป็น 45%
   * โรงเรียนควรจัดทำระบบคัดกรองเด็กที่มี `failures >= 1` หรือเด็กที่ไม่มีเป้าหมายเรียนต่อ เพื่อส่งครูที่ปรึกษาเข้าไปประกบตั้งแต่เดือนแรก

---

## 🚀 6. วิธีการติดตั้งและรันโปรเจกต์ (Quickstart)

```bash
# 1. Clone repository นี้มายังเครื่อง
git clone https://github.com/<USERNAME>/student-grade-prediction.git
cd student-grade-prediction

# 2. ติดตั้ง Dependencies ที่จำเป็น
pip install -r requirements.txt

# 3. เปิดใช้งานผ่าน Jupyter Notebook หรือ JupyterLab
jupyter notebook
```
ลำดับการรันสมุดบันทึก:
1. `notebooks/01_eda_and_data_cleaning.ipynb` $\rightarrow$ เพื่อเตรียมข้อมูลและสร้าง `train.csv`, `test.csv`
2. `notebooks/02_decision_tree_model.ipynb` $\rightarrow$ เพื่อเทรนและประเมิน Decision Tree
3. `notebooks/03_naive_bayes_model.ipynb` $\rightarrow$ เพื่อเทรนและประเมิน Naive Bayes
4. `notebooks/04_model_comparison.ipynb` $\rightarrow$ เพื่อเปรียบเทียบผลลัพธ์ของทั้งสองโมเดล
