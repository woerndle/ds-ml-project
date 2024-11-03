# todos exercise 1
- [ ] Pick 3 Classifier => maybe split work on classifier level, then bottleneck is preprocessing
  - [ ] pick cross classifier performance metric for comparison
[ ] Introduce the 4 Datasets => same for all experiments
[ ] Preprocessing for the 4 Datasets => same for all experiments?
[ ] For each Classifier:
- [ ] evaluate different model parameters and pre-processing-mthods on performance for the different types of datasets
- [ ] compare holdout to cross-validation => maybe random holdout vs distributed holdout vs cross-validation
- [ ] pattern/trend analysis:
    - [ ] Which methods work well and which did not, is there e.g. one method outperforming the others on all datasets?
    - [ ] How do the results change when preprocessing strategies change? 
    - [ ] How sensitive is an algorithm to parameter settings?
    - [ ] Are there differences across the datasets? Design your experiments so that you can investigate the influence of single parameters.

# experiments preprocessing
Different relevance of features => interesting what is the single most important feature per classifier dataset combination? <br>
training with only one feature, given a classifier and HP combination and then compare results for different scaling
| Feature | MinMax | Standard | Robust |
| ------- | -----: | -------: | -----: |
| F1      |   1600 |     1600 |   1600 |
| F2      |     12 |       12 |     12 |
| F3      |      1 |        1 |      1 |
Classifier xyz Dataset abc Scaling s (https://eitca.org/artificial-intelligence/eitc-ai-mlp-machine-learning-with-python/regression/pickling-and-scaling/examination-review-pickling-and-scaling/what-are-some-common-scaling-techniques-available-in-python-and-how-can-they-be-applied-using-the-scikit-learn-library/#:~:text=These%20techniques%20include%20standardization%2C%20min,max%20scaling%2C%20and%20robust%20scaling.&text=In%20addition%20to%20these%20common,power%20transformation%20and%20quantile%20transformation.) or (https://www.geeksforgeeks.org/scaling-techniques-in-machine-learning/)
| s        | accuracy-holdout-random | accuracy-holdout-distributed | accuracy-cross-validation |
| -------- | ----------------------: | ---------------------------: | ------------------------: |
| MinMax   |                    1600 |                         1600 |                      1600 |
| Standard |                      12 |                           12 |                        12 |
| Robust   |                       1 |                            1 |                         1 |

# experiments holdout vs cross-validation
Classifier xyz: => 3 Tables
| dataset              | accuracy-holdout-random | training-time-holdout-random |
| -------------------- | ----------------------: | ---------------------------: |
| amazon-review        |                    1600 |                         1600 |
| congressional-voting |                      12 |                           12 |
| tehran-housing       |                       1 |                            1 |
| wine-reviews         |                       1 |                            1 |

Columns:
- prediction-accuracy-holdout-random
- prediction-accuracy-holdout-distributed
- prediction-accuracy-cross-validation
- training-time-holdout-random
- training-time-holdout-distributed
- training-time-cross-validation

# experiments hyperparameter (HP)
N-Diagrams for each HP, where N is the number of hyperparameters of that classifier <br>
Classifier xyz Dataset abc Parameter p
| p    | accuracy-holdout-random | accuracy-holdout-distributed | accuracy-cross-validation |
| ---- | ----------------------: | ---------------------------: | ------------------------: |
| -Inf |                    1600 |                         1600 |                      1600 |
| 0    |                      12 |                           12 |                        12 |
| Inf  |                       1 |                            1 |                         1 |


