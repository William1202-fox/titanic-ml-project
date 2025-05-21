# titanic-ml-project
End-to-end machine–learning pipeline for Kaggle **Titanic: Machine Learning from Disaster** competition.

> **Public LB score :** **0.76076** (Rank ≈ 13 k / 40 k)

---

## 1 ．Problem statement
Predict whether a passenger survived the Titanic sinking based on demographic and ticket information.  
二元分類問題，官方評分指標為 **Accuracy**。

---

## 2 ．Data overview
| File | Rows × Cols | Notes |
|------|-------------|-------|
| `train.csv` | 891 × 12 | 含 `Survived` 標籤 |
| `test.csv`  | 418 × 11 | 不含標籤，用於提交 |
| 缺值情況 | `Cabin` 77 %、`Age` 20 %、`Embarked` < 1 % |

主要特徵洞見：
* **Sex** – female survival ≈ 74 %，male ≈ 19 %
* **Pclass** – 1st class survival ≈ 63 %，3rd class ≈ 24 %
* `FamilySize = SibSp + Parch + 1`：大型家庭 (≥ 5) 生存率最低

---

## 3 ．Pipeline
1. **EDA** – 缺值檢查、數值分布、類別比例  
2. **Feature engineering**  
   * Drop `Cabin`, `Ticket`, `Name`  
   * Add `FamilySize`, `IsAlone`  
3. **Pre-processing** –  
   * `SimpleImputer` (median / most-frequent)  
   * `OneHotEncoder` for categorical features  
   * Wrapped with `ColumnTransformer`  
4. **Models**  
   | Model | CV Accuracy | Notes |
   |-------|------------|-------|
   | Logistic Regression | 0.790 ± 0.023 | baseline |
   | Random Forest (default) | 0.807 ± 0.028 | n = 400 trees |
   | **Random Forest (tuned)** | **0.828 ± 0.018** | best params below |

Best params  
```yaml
n_estimators: 400
max_depth: None
min_samples_split: 2
min_samples_leaf: 1
random_state: 42

