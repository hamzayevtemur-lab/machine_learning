# 1. What is a Decision Tree, and how does it make predictions?
# A Decision Tree is a model that splits data based on feature conditions (e.g., age > 30).
# It makes predictions by following a path from the root to a leaf node, where the final prediction is stored.

# 2. What does it mean for a node to be pure or impure?
# Pure node: All samples belong to the same class.
# Impure node: Samples belong to multiple classes.

# 3. Role of Entropy and Gini impurity
# They measure how impure a node is that
# Entropy: Measures randomness/disorder.
# Gini impurity: Measures probability of misclassification.Used to decide the best split.

# 4. What is Information Gain, and why is it used?
# Information Gain = reduction in impurity after a split.
# It is used to choose the split that best separates the data.

# 5. Why are Decision Trees greedy algorithms?
# They choose the best split at each step locally, without considering future splits.

# 6. Why do deep trees overfit?
# They memorize noise and details of training data, leading to poor generalization.

# 7. Bias–variance tradeoff
# Shallow tree: High bias, low variance (underfits)
# Deep tree: Low bias, high variance (overfits)

# 8. What is pruning, and why is it necessary?
# Pruning removes unnecessary branches to reduce overfitting and improve generalization.

# 9. Why no feature scaling is needed?
# Decision Trees split based on feature thresholds, not distances, so scaling doesn’t affect them.

# 10. One advantage and one limitation
# Advantage: Easy to understand and interpret
# Limitation: Prone to overfitting (especially deep trees)