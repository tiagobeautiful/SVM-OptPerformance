# SVM-OptPerformance
Algorithms implemented for the submmitted paper "Performance of optimization methods applied to the Support Vector Machine training problem".
Paper: <soon>

## Abstract
Support Vector Machines (SVM) have proven to be one of the most useful approaches for Machine Learning, being applied in various strategic areas. On the other hand, the theory of nonlinear Optimization plays an important role in SVM, especially in training its model. This work presents a discussion on the possibilities of nonlinear Optimization algorithms for SVM training, such as Projection methods, Interior Point methods, Active Constraints and Filter methods. These algorithms were implemented in MATLAB\textregistered, and numerical experiments were conducted with datasets generated randomly and from Machine Learning repositories. From the experiments performed, considering the quality of the solutions found, training performance profiles, as well as observing predictive metrics such as accuracy, F1 Score, and Matthews Correlation Coefficient, the results indicate that even a naive implementation of the Sequential Minimal Optimization (SMO) was more efficient in most criteria when compared to the other implemented algorithms.

## Considerations
Two applications were considered, one based on randomly generated datasets and another based on benchmark datasets extracted from Machine Learning repositories. The obtained results were discussed and analyzed from the perspective of training performance and generalization ability of the generated models, and what became evident is a better performance of the Spectral Projected Gradient with Augmented Lagrangian, Filter methods, quadprog function (using Interior Point algorithm) and SMO based methods (using Platt's Maximum Violating Pair or Fan et al Maximum Violating Pair using second order information), not necessarily in this order. However, the evaluated training time for most tested methods was the main disadvantage compared to Active Set methods. For the conducted numerical experiments, linear and RBF kernel functions were implemented to develop classification models that allow modeling the relationships between input data (linear or non-linear). In the obtained results, the main difference in using these two functions was regarding the computational cost for training. These results highlight the importance of carefully choosing the most appropriate Optimization method for training SVM, considering both the nature of the data and the expected performance goals. Overall, the projection methods did not perform well with fixed method hyperparameters, except for the Spectral Projected Gradient method. The methods involving Filter and Interior Points achieved good classification metrics but were slower in training. The methods based on Active Set, emulating Platt and Fan papers, have a better balance between training speed and classification metrics. Thus, the results indicate that the basic implementation of SMO-based algorithms is more efficient than the other Optimization classes implemented here.

## Further analysis of the numerical experiments
[Download PDF](./_results/SurvivalAnalysis_and_PCA-Analysis.pdf)
