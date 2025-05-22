#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


# In[70]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.head()


# In[71]:


train_df.drop(columns=['Unnamed: 0'],inplace=True)


# In[72]:


test_df.drop(columns=['Unnamed'], inplace=True)


# In[73]:


print(test_df.columns.tolist())


# In[74]:


print(train_df.columns.tolist())


# In[75]:


train_df.columns = train_df.columns.str.lower()


# In[76]:


test_df.columns = test_df.columns.str.lower()


# In[77]:


X_train = train_df.drop('event', axis=1)
y_train = train_df['event']

X_test = test_df.drop('event', axis=1)
y_test = test_df['event']


# ## Standardize the feature inputs

# In[78]:


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)  # Fit and transform on train
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)        # Only transform on test


# In[79]:


start_time = time.time()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

end_time = time.time()
training_time= end_time-start_time

print(f"✅ Model training complete in {training_time:.4f} seconds")


# In[80]:


start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()
inference_time= end_time-start_time

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Prediction complete in {inference_time:.4f} seconds")


# In[81]:


model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


# In[82]:


y_pred = model.predict(X_test)

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Evaluation complete.")


# In[83]:


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


# In[84]:


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


# In[85]:


from IPython.display import Image
Image(filename='decision_tree_full.png')


# In[86]:


import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define objective function
def objective(trial):
    # Define the hyperparameters to search
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create and train the model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on test set
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Run optimization (50 trials is good for testing)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Show best result
print("Best Hyperparameters:", study.best_params)


# In[87]:


# Train the best model using found parameters
best_params = study.best_params

start_time = time.time()

best_model = DecisionTreeClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

end_time = time.time()
Training_time= end_time-start_time
print(f"Training complete in {training_time:.4f} seconds")


# In[88]:


# Evaluate
start_time = time.time()

y_pred = best_model.predict(X_test)

end_time = time.time()
Inference_time= end_time-start_time

print(f"Prediction complete in {inference_time:.4f} seconds")


# In[89]:


print(classification_report(y_test, y_pred,output_dict=True))


# In[90]:


y_pred = best_model.predict(X_test)

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[98]:


shap_values = shap.TreeExplainer(best_model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[ ]:




