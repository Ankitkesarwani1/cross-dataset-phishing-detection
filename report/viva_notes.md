# Viva Voce & Defense Preparation Guide
## Cross-Dataset Phishing Website Detection Using Ensemble Learning and Explainable Artificial Intelligence

---

### Q1: What is the primary research objective and novelty of this project?
**Answer:**
Most phishing detection literature tests models on the same dataset from which training samples were drawn, reporting inflated accuracies (>99%). Our research addresses the core question: **"Do phishing detection models generalize across distinct distributions, or do they merely memorize dataset-specific feature distributions?"**
We evaluate five diverse classifiers across two benchmark datasets (PhiUSIIL URL Dataset and UCI Phishing Websites) under both in-dataset and cross-dataset testing paradigms after rigorous feature alignment.

---

### Q2: Why is testing on a separate dataset (cross-dataset evaluation) necessary in cyber threat intelligence?
**Answer:**
Phishing tactics evolve dynamically. Attackers change URL structures, obfuscation techniques, and hosting infrastructure. Evaluating models on unseen data collected from a different source and timeframe tests actual generalization against distributional shift. If a classifier achieves 99% in-dataset but drops to 50–70% cross-dataset, it proves that the model overfitted to specific sampling biases rather than learning invariant signatures of phishing.

---

### Q3: What were the two datasets used, and how do they differ?
**Answer:**
1. **PhiUSIIL Phishing URL Dataset (UCI id 967):**
   - 235,795 instances, 54 features, published in 2023.
   - Features include continuous URL character counts, lexical entropy, and webpage structure metrics. Original target: 0 = phishing, 1 = legitimate.
2. **UCI Phishing Websites Dataset (UCI id 327):**
   - 11,055 instances, 30 discrete categorical features, published in 2014 by Mohammad et al.
   - All features are trinary-encoded ({-1, 0, 1}). Original target: -1 = phishing, 1 = legitimate.

---

### Q4: How did you harmonize the target labels across both datasets?
**Answer:**
We created a uniform target column `is_phishing`:
- Phishing = `1` (positive class of interest in security systems).
- Legitimate = `0` (negative baseline).
For PhiUSIIL, raw label `0` was mapped to `1`, and `1` to `0`. For UCI, raw result `-1` was mapped to `1`, and `1` to `0`.

---

### Q5: Why did you drop text and identifier columns (e.g., URL string, Domain, TLD, Title) before modeling?
**Answer:**
Including raw text identifiers like `FILENAME`, `URL`, `Domain`, or `Title` allows models to memorize specific domain names (e.g., "paypal-login-secure.com") or top-level domains. High-capacity learners like Random Forests or XGBoost would memorize these string tokens, resulting in perfect in-dataset metrics that catastrophically fail in real-world deployment on new domains. Dropping them forces models to learn from structural, syntactic, and behavioral patterns.

---

### Q6: What duplicate records or conflicting feature vectors did you uncover during preprocessing?
**Answer:**
- **PhiUSIIL:** 808 exact duplicate records were identified and dropped (0 conflicting feature vectors).
- **UCI Phishing Websites:** 5,206 exact duplicate rows were detected (leaving 5,849 unique samples). Crucially, we discovered **64 conflicting feature vectors** (identical feature inputs assigned to both legitimate and phishing labels). This highlights inherent label noise in the original 2014 UCI dataset, setting an theoretical upper bound on accuracy.

---

### Q7: Explain the feature alignment methodology. Why are there only 9 common features?
**Answer:**
PhiUSIIL and UCI measure different aspects of websites. Cross-dataset evaluation requires an exact 1-to-1 semantic equivalence. By inspecting the actual dataset schemas and original academic documentation (Mohammad et al., 2014; Prasad et al., 2023), we identified **9 semantically equivalent features**:
1. `IP_in_URL`
2. `URL_Length`
3. `Sub_Domain`
4. `SSL_State`
5. `Favicon`
6. `Iframe`
7. `Popup`
8. `Redirect`
9. `Has_External_FormSubmit` (SFH)

Continuous PhiUSIIL features were recoded into the UCI {-1, 0, 1} scheme using documented heuristic thresholds.

---

### Q8: What adaptation was made for the `Redirect` feature during alignment?
**Answer:**
In UCI Rule 19, `Redirect` is coded based on redirect count: $\ge 4$ redirects = phishing (-1), 2–3 = suspicious (0), $\le 1$ = legitimate (1). However, PhiUSIIL's feature `NoOfURLRedirect` has a maximum value of 1. Applying a threshold of $\ge 4$ would render the column completely constant (all zeros). We adapted the threshold to $\ge 1$ (any redirect present = 1, absent = 0) and explicitly categorized its confidence of equivalence as "low" in the mapping table.

---

### Q9: Why does PhiUSIIL with all 50 features achieve near 100% accuracy? Is this data leakage?
**Answer:**
PhiUSIIL contains synthetic and deterministic features such as `URLSimilarityIndex`, `URLCharProb`, and `TLDLegitimateProb`. For example, `URLSimilarityIndex` alone yielded a Mutual Information score of **0.661**. An ablation experiment proved that dropping these three top features only lowered Random Forest test accuracy from 99.998% to 99.955%. This indicates that PhiUSIIL contains multiple redundant, deterministic features rather than simple procedural data leakage.

---

### Q10: What is data leakage, and how did your codebase mathematically prevent it?
**Answer:**
Data leakage occurs when information from outside the training dataset (such as the validation or test set) contaminates model fitting. We prevented leakage by:
1. Deduplicating datasets *before* splitting to prevent identical records from existing in both train and test partitions.
2. Using `sklearn.pipeline.Pipeline` with `ColumnTransformer` and `StandardScaler`. Scaling parameters ($\mu, \sigma$) and log-transform boundaries are fit exclusively on the training fold during Stratified 5-Fold Cross-Validation and held-out evaluation.
3. Keeping a strict `random_state=42` across all splits.

---

### Q11: Why did you use `LinearSVC` instead of an RBF kernel SVM on PhiUSIIL?
**Answer:**
Standard kernel SVM (`sklearn.svm.SVC`) has a computational time complexity of $\mathcal{O}(n^2)$ to $\mathcal{O}(n^3)$ with respect to sample size $n$. On PhiUSIIL's 234,987 instances, an RBF kernel would require hours to days of training time. `LinearSVC` uses LIBLINEAR, scaling linearly $\mathcal{O}(n \cdot d)$, allowing convergence in seconds while maintaining competitive linear decision boundaries.

---

### Q12: Why was `CalibratedClassifierCV` used with `LinearSVC`?
**Answer:**
`LinearSVC` optimizes a hinge loss and outputs an uncalibrated signed distance to the hyperplane via `decision_function`, but does not output class probabilities (`predict_proba`). To compute ROC-AUC curves, cross-entropy metrics, and SHAP values consistently across all five models, we wrapped the fitted `LinearSVC` in `CalibratedClassifierCV` using Platt scaling (sigmoid calibration).

---

### Q13: What were the key findings in the baseline experiments (Exp 1 and Exp 2)?
**Answer:**
- **Exp 1 (PhiUSIIL all features):** Near perfect performance ($F_1 \approx 1.000, \text{AUC} \approx 1.000$).
- **Exp 1 (PhiUSIIL aligned 9 features):** Performance drops to realistic levels ($F_1 = 0.828 - 0.863, \text{AUC} \approx 0.959$).
- **Exp 2 (UCI all 30 features):** XGBoost led with $F_1 = 0.951, \text{AUC} = 0.985$.
- **Exp 2 (UCI aligned 9 features):** Random Forest and XGBoost achieved $F_1 \approx 0.880, \text{AUC} \approx 0.934$.
This confirms that the 9 aligned features capture substantial phishing signal, but discard dataset-specific shortcuts.

---

### Q14: What happened in Experiment 3 (Train on PhiUSIIL Aligned $\to$ Test on UCI)?
**Answer:**
Models experienced severe performance degradation:
- Test Accuracy dropped from ~87% down to **58% - 59%**.
- Precision dropped to **~0.56**, while Recall was extremely high (**~0.97 - 0.98**).
- The models over-predicted the phishing class (**~90% predicted phishing rate** vs. the actual **51.6%** prevalence).
- Tree models and ensembles overfitted to PhiUSIIL's decision boundaries; Logistic Regression proved most resilient.

---

### Q15: What happened in Experiment 4 (Train on UCI Aligned $\to$ Test on PhiUSIIL)?
**Answer:**
Models exhibited an inverted asymmetry:
- Test Accuracy was **71% - 79%**, but Recall collapsed to **35% - 51%** (high false negative rate).
- Precision was exceptionally high (**91% - 100%**).
- Predicted phishing rate was only **16% - 22%** compared to the true base rate of **42.6%**.
- Models learned strict rules from the older UCI dataset and failed to recognize modern phishing instances in PhiUSIIL.

---

### Q16: Why did models suffer such a severe drop in cross-dataset evaluation?
**Answer:**
This drop is caused by **covariate shift** and **concept drift**:
1. **Temporal Gap:** UCI was compiled in 2014; PhiUSIIL was compiled in 2023. Attack methodologies changed over this 9-year span (e.g., ubiquitous HTTPS adoption on phishing sites).
2. **Measurement Differences:** Subdomain counting, favicon loading, and iframe injection are collected under slightly different browser contexts and threshold heuristics.
3. **Class Priors & Threshold Bias:** Default classification thresholds ($P \ge 0.5$) calibrated to one distribution shift when applied to another.

---

### Q17: Which model demonstrated the highest resilience to cross-dataset transfer?
**Answer:**
**Logistic Regression** demonstrated the most robust cross-dataset generalization ($F_1 = 0.707$ in Exp 3 and $F_1 = 0.676$ in Exp 4). While non-linear ensemble models (Random Forest, XGBoost) won comfortably within the same dataset, their complex partitioned decision boundaries overfitted to dataset-specific feature distributions. Linear models learned monotonic relationships that transferred better.

---

### Q18: What is SHAP, and why did you use `TreeExplainer`?
**Answer:**
SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It computes the marginal contribution of each feature to the prediction for every sample, satisfying desirable theoretical properties: local accuracy, missingness, and consistency. We used `shap.TreeExplainer`, which computes exact tree SHAP values in polynomial time $\mathcal{O}(TLD^2)$ rather than exponential model-agnostic sampling.

---

### Q19: What did the SHAP importance analysis and Spearman rank correlation reveal?
**Answer:**
- In both datasets, `SSL_State`, `URL_Length`, and `Sub_Domain` emerged as dominant drivers of predictions.
- However, the relative weight and ranking of secondary features (`Has_External_FormSubmit`, `Favicon`, `Iframe`) varied substantially between the two datasets.
- The Spearman rank correlation ($\rho$) of feature importances between models confirmed moderate agreement, explaining why models transferred partially but suffered from distributional misalignment.

---

### Q20: What are three concrete recommendations for deploying phishing detection systems in industry?
**Answer:**
1. **Continuous Domain Adaptation:** Never deploy an off-the-shelf static classifier; retrain periodically on live traffic streams using active learning.
2. **Dynamic Threshold Tuning:** Do not rely on default $0.5$ classification cutoffs. Adjust the threshold based on the operational cost ratio between False Positives (user friction) and False Negatives (breach).
3. **Hybrid Feature Engineering:** Combine lightweight URL lexical features for line-rate screening with asynchronous deep page-content analysis and certificate transparency logs.

