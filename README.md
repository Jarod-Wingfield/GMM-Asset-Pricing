# GMM Estimation for Multifactor Asset Pricing Models

This repository implements **two-stage Generalized Method of Moments (GMM)** estimation and testing for multifactor asset pricing models. I test by using **Fama-French 5 factors** and **25 portfolios (5×5)** as test assets to analyze **price of risk, pricing errors, and risk exposures**.

It includes:

* Two-stage **GMM estimation** with optional Newey-West adjustments
* Linear Factor Models in **Discount Factor Form** with excess returns
* **Cross-sectional regressions** for risk premia (λ) and pricing errors (α)
* Computation of **SSQE**, **MAPE**, and **J-statistic**

---

## Repository Structure

* **`GMM_SDF.py`**
  Implements the **GMM\_Two\_Stage** class, including functions for loading data, estimating factor prices of risk, pricing errors, and cross-sectional regressions.

* **`data`**
  Contains **test asset returns** (25 portfolios) and **Fama-French 5-factor data** sourced from [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

* **`GMM_SDF_main.ipynb`**
  Example Jupyter notebooks demonstrating GMM estimation, risk exposure analysis, and table generation.

---

## References

* **Cochrane, J. H. (2009).** *Asset Pricing: Revised Edition.* Princeton University Press.
* **Hsu, P. H., Li, K., & Tsou, C. Y. (2023).** *The Pollution Premium.* The Journal of Finance, 78(3), 1343–1392.
* **Burnside, C. (2011).** *The Cross Section of Foreign Currency Risk Premia and Consumption Growth Risk: Comment.* American Economic Review, 101(7), 3456–3476.
* **Belo, F., Li, J., Lin, X., & Zhao, X. (2017).** *Labor-Force Heterogeneity and Asset Prices: The Importance of Skilled Labor.* The Review of Financial Studies, 30(10), 3669–3709.
* **Bae, J. W., Bali, T. G., Sharifkhani, A., & Zhao, X. (2022).** *Labor Market Networks, Fundamentals, and Stock Returns.* Georgetown McDonough School of Business Research Paper, (3951333).

**Online Resources:**

* [GMM Estimation and Testing of Multifactor Asset Pricing Models – Quant StackExchange](https://quant.stackexchange.com/questions/75783/r-resources-for-gmm-estimation-and-testing-of-multifactor-asset-pricing-models)
* [GMM Implementation in Asset Pricing – Zhihu](https://zhuanlan.zhihu.com/p/92490417)

---

## 📄 License

MIT License. See `LICENSE` file for details.
