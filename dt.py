#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


# In[59]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.head()


# In[60]:


train_df.drop(columns=['Unnamed: 0'],inplace=True)


# In[61]:


test_df.drop(columns=['Unnamed'], inplace=True)


# In[62]:


print(test_df.columns.tolist())


# In[63]:


print(train_df.columns.tolist())


# In[64]:


train_df.columns = train_df.columns.str.lower()


# In[65]:


test_df.columns = test_df.columns.str.lower()


# In[66]:


X_train = train_df.drop('event', axis=1)
y_train = train_df['event']

X_test = test_df.drop('event', axis=1)
y_test = test_df['event']


# ## Standardize the feature inputs

# In[67]:


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)  # Fit and transform on train
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)        # Only transform on test


# In[68]:


start_time = time.time()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

end_time = time.time()
training_time= end_time-start_time

print(f"✅ Model training complete in {training_time:.4f} seconds")


# In[69]:


start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
inference_time= end_time-start_time

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Prediction complete in {inference_time:.4f} seconds")


# In[70]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Set the figure size for better readability
plt.figure(figsize=(20, 10))

# Plot the tree
plot_tree(
    model,                      # your trained DecisionTreeClassifier
    feature_names=X_train.columns,  # use actual column names
    class_names=['No Event', 'Event'],  # labels for target classes
    filled=True,                # color nodes by class
    rounded=True,               # rounded boxes
    max_depth=3,                # limit the depth to avoid clutter (adjust as needed)
    fontsize=10
)

plt.title("Decision Tree Visualization (First 3 Levels)")
plt.show()


# In[71]:


from sklearn.tree import export_graphviz
import graphviz

# Export the decision tree as a DOT format string
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X_train.columns,
    class_names=['No Event', 'Event'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Render and save as PNG
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision_tree_full")  # Will create decision_tree_full.png


# In[72]:


from IPython.display import Image
Image(filename='decision_tree_full.png')


# In[73]:


from bayes_opt import BayesianOptimization
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Objective function to maximize (higher F1 score = better model)
def optimize_decision_tree(max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    # Use cross-validation score (f1 macro for imbalanced data)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
    return np.mean(scores)

# Define the hyperparameter search space
param_bounds = {
    'max_depth': (3, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Initialize the Bayesian optimizer
optimizer = BayesianOptimization(
    f=optimize_decision_tree,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)

# Start optimization
optimizer.maximize(init_points=5, n_iter=20)

# Best parameters found
print("\nBest parameters found:")
print(optimizer.max)

# Store best parameters found by BayesianOptimization
best_params = optimizer.max['params']


# In[74]:


# Extract and convert best parameters
raw_best_params = optimizer.max['params']
best_params = {
    'max_depth': int(raw_best_params['max_depth']),
    'min_samples_split': int(raw_best_params['min_samples_split']),
    'min_samples_leaf': int(raw_best_params['min_samples_leaf'])
}

# Train the best model using found parameters
from sklearn.tree import DecisionTreeClassifier
import time

start_time = time.time()

best_model = DecisionTreeClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

print(f"✅ Training complete in {training_time:.4f} seconds")


# In[75]:


# Evaluate
start_time = time.time()

y_pred = best_model.predict(X_test)

end_time = time.time()
Inference_time= end_time-start_time

print(f"Prediction complete in {inference_time:.4f} seconds")


# In[76]:


print(classification_report(y_test, y_pred,output_dict=True))


# In[81]:


print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[80]:


print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))


# In[77]:


y_pred = best_model.predict(X_test)

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[78]:


shap_values = shap.TreeExplainer(best_model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[79]:


import lime
import lime.lime_tabular
import numpy as np

# Create the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),        # Your training features
    feature_names=X_train.columns,          # Column names
    class_names=['No Event', 'Event'],      # Target class names
    mode='classification'                   # Since you're predicting classes
)

# Choose an instance from the test set to explain
i = 10  # You can change this index to any row in your test set
exp = explainer.explain_instance(
    data_row=X_test.iloc[i],
    predict_fn=model.predict_proba
)

# Show explanation in notebook
exp.show_in_notebook(show_table=True)

