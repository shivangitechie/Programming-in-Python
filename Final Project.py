#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering



my_missing = ['NA','NULL','-999','-1'] # Load the values in the list as missing values
male_stud = pd.read_csv("male_stud.csv", na_values = my_missing)



print(f"The total number of male students whose details were recorded is {male_stud.shape[0]}") # Checks the number of students included in the dataset
print(f"The total number of indicator variables is {male_stud.shape[1]-1}") # Checks the number of indicator variables excluding the target variable
print("The number of missing values in each column")
print(male_stud.isnull().sum()) # Prints the total number of null values in each column
male_stud.head() 


# For numerical summary of the data
male_stud.describe()


correlation_matrix = male_stud.corr()
correlation_matrix # Prints the correlation matrix


print(male_stud.studytime.unique()) 
print(f'The total number of unique values of Studytime are {len(male_stud.studytime.unique())}') 

print(male_stud.failures.unique())
print(f'The total number of unique values of Failures are {len(male_stud.failures.unique())}')

print(male_stud.famrel.unique())
print(f'The total number of unique values of Family Relations are {len(male_stud.famrel.unique())}')

print(male_stud.goout.unique())
print(f'The total number of unique values of Going out are {len(male_stud.goout.unique())}')


print("Value count for studytime")
print(male_stud.studytime.value_counts())

print("Value count for failures")
print(male_stud.failures.value_counts())

print("Value count for famrel")
print(male_stud.famrel.value_counts())

print("Value count for goout")
print(male_stud.goout.value_counts())


print("Value count for large_family")
print(male_stud.large_family.value_counts())

print("Value count for lives_in_city")
print(male_stud.lives_in_city.value_counts())

print("Value count for paid")
print(male_stud.paid.value_counts())

print("Value count for activities")
print(male_stud.activities.value_counts())

print("Value count for internet")
print(male_stud.internet.value_counts())

print("Value count for romantic")
print(male_stud.romantic.value_counts())


sns.set(style="whitegrid")

plt.figure(figsize=(15, 10))

# Plot boxplots
plt.subplot(2, 3, 1)
sns.boxplot(x='studytime', y='final_grade', data=male_stud, palette="Set3")
plt.title('Boxplot of Final Grade by Study Time')
plt.xlabel('Study Time')
plt.ylabel('Final Grade')

plt.subplot(2, 3, 2)
sns.boxplot(x='traveltime', y='final_grade', data=male_stud, palette="Set3")
plt.title('Boxplot of Final Grade by Travel Time')
plt.xlabel('Travel Time')
plt.ylabel('Final Grade')

plt.subplot(2, 3, 3)
sns.boxplot(x='failures', y='final_grade', data=male_stud, palette="Set3")
plt.title('Boxplot of Final Grade by Failures')
plt.xlabel('Failures')
plt.ylabel('Final Grade')

plt.subplot(2, 3, 4)
sns.boxplot(x='famrel', y='final_grade', data=male_stud, palette="Set3")
plt.title('Boxplot of Final Grade by Family Relationship')
plt.xlabel('Family Relationship')
plt.ylabel('Final Grade')

plt.subplot(2, 3, 5)
sns.boxplot(x='goout', y='final_grade', data=male_stud, palette="Set3")
plt.title('Boxplot of Final Grade by Going Out')
plt.xlabel('Going Out')
plt.ylabel('Final Grade')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()



plt.figure(figsize=(10, 6))

male_stud.absences.hist(bins=20,density=True,color='m',alpha=0.9)

plt.title('Distribution of Absences for Male Students')
plt.xlabel('Absences')
plt.ylabel('Density')
# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


plt.figure(figsize=(10, 6))

male_stud.final_grade.hist(bins=20,density=True,color='m',alpha=0.9)

plt.title('Distribution of Final Grade for Male Students')
plt.xlabel('Final Grade')
plt.ylabel('Density')
# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

cat_bin_cols = ['studytime', 'traveltime', 'failures', 'famrel', 'goout', 'large_family', 'lives_in_city', 'paid', 'activities', 'internet', 'romantic']

# Set up the subplot grid
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.subplots_adjust(hspace=0.5)

# Flatten the 2D array of axes for easier iteration
axes = axes.flatten()

# Loop through each categorical variable
for i, col in enumerate(cat_bin_cols):
    counts = male_stud[col].value_counts()

    # Use a different color for each plot
    colors = sns.color_palette("husl", len(counts))

    # Plot the bar chart
    counts.plot(kind='bar', color=colors, alpha=0.9, ax=axes[i])

    # Set titles and labels
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Hide any remaining empty subplots
for j in range(len(cat_bin_cols), len(axes)):
    fig.delaxes(axes[j])

# Show the plots
plt.show()



# Let us visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

female_stud = pd.read_csv("female_stud.csv", na_values = my_missing) # Load the female students dataset


female_stud.head()


print(f"The total number of female students whose details were recorded is {female_stud.shape[0]}")
print(f"The total number of indicator variables is {female_stud.shape[1]-1}")
print("The number of missing values in each column")
print(female_stud.isnull().sum())
female_stud.head()


# Set alpha
alpha = 0.01 # given

columns = male_stud.columns

for col in columns:
    t_stat, p_value = stats.ttest_ind(male_stud[col], female_stud[col]) # equal_var=False)  # Assuming unequal variances
    print(f"\nT-test results for {col}:")
    print(f"T-score: {t_stat}")
    print(f"P-value: {p_value}")


students = pd.concat([male_stud,female_stud])
students

column_names = ['column1','column2','correlation','pvalue']
correlation_pairs = pd.DataFrame(columns=column_names) # To store the columns and their corresponding correlation and p-value

for i in columns: # The columns variable is the list declared for ttest above which excludes the target variable
    for j in columns:
        if(i != j): # When the 2 columns are not the same find their pearsonr coefficient
            corr_coeff, p_value = stats.pearsonr(students[i], students[j])  # Assuming unequal variances
            correlation_pairs = correlation_pairs.append({
                'column1': i,
                'column2': j,
                'correlation': corr_coeff,
                'pvalue': p_value
            }, ignore_index=True)

# Get the absolute value of correlation between 2 indicator variables
correlation_pairs.correlation = abs(correlation_pairs.correlation) 
# Sort the correlation_pairs dataframe based on absolute value of correlation
correlation_pairs = correlation_pairs.sort_values(by='correlation', ascending=False) 
correlation_pairs = correlation_pairs.reset_index(drop=True)
correlation_pairs
print("\nTop 4 most correlated pairs:")
print(correlation_pairs.head(8))


# Let us plot the top 4 most correlated pairs to visualize their correlation
plt.figure(figsize=(15, 10))

plt.subplot(2,2,1)
sns.stripplot(data=students, x=students.failures, y=students.final_grade, jitter = True, alpha = 0.9)
plt.title(f'Scatter Plot: final_grade vs failures')
plt.xlabel('failures')
plt.ylabel('final_grade')

plt.subplot(2,2,2)
sns.stripplot(data=students, x=students.traveltime, y=students.lives_in_city, jitter = True, alpha = 0.9)
plt.title(f'Scatter Plot: lives_in_city vs traveltime')
plt.xlabel('traveltime')
plt.ylabel('lives_in_city')
#plt.show()

plt.subplot(2,2,3)
sns.stripplot(data=students, x=students.goout, y=students.freetime, jitter = True, alpha = 0.9)
plt.title(f'Scatter Plot: freetime vs goout')
plt.xlabel('goout')
plt.ylabel('freetime')

plt.subplot(2,2,4)
sns.stripplot(data=students, x=students.lives_in_city, y=students.internet, jitter = True, alpha = 0.9)
plt.title(f'Scatter Plot: internet vs lives_in_city')
plt.xlabel('lives_in_city')
plt.ylabel('internet')

plt.show()

students['pass_fail'] = np.where(students['final_grade'] >= 10, 1, 0) # Create the column pass_fail
# 1 signifies PASS
# 0 signifies FAIL
print(sum(students['pass_fail']==1)) # Number of students passed
print(sum(students['pass_fail']==0)) # Number of students failed


students.tail(10)

# Predictor variables
X = students.drop(['final_grade','pass_fail'],axis=1)
X_std = (X-X.mean())/X.std()
# Add intercept
X_std.insert(0,'intercept',1)

# Target variable
y = students.pass_fail 


X_std.head()


logit1 = sm.Logit(y, X_std).fit() # Fit the logistic regression model
print(logit1.summary())

def forward_selection(X, y, criterion='AIC'):
    selected_features = [] # Features that reduce the aic
    left_features = list(X.columns) # remaining features
    best_aic = np.inf # set the worst value of aic

    while True:
        aic_values = [] # store aic values
        for feature in left_features: # Add a feature one by one
            model = sm.Logit(y, X[selected_features + [feature]]) # fit the model with the selected features + 1 feature
            result = model.fit(disp=False)
            aic = result.aic
            aic_values.append((feature, aic)) # Update the aic values

        aic_values.sort(key=lambda x: x[1])
        best_feature, best_feature_aic = aic_values[0] # Get the best aic value and the corresponding feature from the list

        if best_feature_aic < best_aic: # If the new aic value is less than the best aic value
            selected_features.append(best_feature) # Add the new feature into the selected features
            left_features.remove(best_feature) # Remove the added feature from the remaining features
            best_aic = best_feature_aic # Update the best aic value
        else: # If the new aic did not reduce the best aic, exit the loop and fit the final model
            break

    final_model = sm.Logit(y,X[selected_features]) # Fit the final model
    result = final_model.fit()
    
    return result, selected_features

# Perform forward selection
result_forward, selected_features_forward = forward_selection(X_std, y)


print("Selected Features:", selected_features_forward)
print(result_forward.summary()) 


# Let us get the original dataset by removing the pass_fail column
original_data = students.drop(['pass_fail'], axis = 1)

# Create the X (indicator) and y (target) variables
X_raw = original_data.drop(['final_grade'],axis=1) # The original non-standardized data (indicators)
y_raw = original_data.final_grade # Target variable

# We will split the data into training and test sets by 75% and 25% respectively.
train_size = np.floor(0.75*X_raw.shape[0]).astype(int) 
np.random.seed(123)
train_select = np.random.permutation(range(len(y_raw)))
X_train = X_raw.iloc[train_select[:train_size],:].reset_index(drop=True)
X_test = X_raw.iloc[train_select[train_size:],:].reset_index(drop=True)
y_train = y_raw.iloc[train_select[:train_size]].reset_index(drop=True)
y_test = y_raw.iloc[train_select[train_size:]].reset_index(drop=True) 


# (b) Fit a random forest regression model with 10 trees using the training data. Include the argument random_state=101 in the random forest regression function to ensure reproducible results. Determine which variables are most important in predicting the final grade of a student. Discuss your findings in relation to the logistic models fit in question 4.

# In[29]:


# Fit the Random Forest Regression Model
rf = RandomForestRegressor(n_estimators=10,random_state = 101)
rf.fit(X_train, y_train)

# To get the importance of features used in the model, we use the feature_importances_ method
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({
   'Feature': X_train.columns,
   'Importance': feature_importances
})
importance_df


## Predict the test results
rf_test_pred = rf.predict(X_test)
## Calculate the RSS value for checking the accuracy of the model
RSS_rf = np.mean(pow((rf_test_pred - y_test),2)) 
print(RSS_rf)

# Plot the predicted values against the true values
fig = plt.figure()
plt.plot(y_test,rf_test_pred,'kx')
plt.plot([0,20], [0,20], ls="--")


# Number of trees to evaluate
num_trees_list = [5, 10, 50, 100, 500, 1000, 5000]

# Store mean and standard error for each number of trees
mean_scores = []
std_scores = []

for num_trees in num_trees_list:
    rf_scores = []
    for i in range(20): # Repeat the model fit and predict 20 times for each number of trees with different random states
        rf_model = RandomForestRegressor(n_estimators=num_trees, random_state=np.random.randint(100))
        rf_model.fit(X_train,y_train)
        rf_test_pred = rf_model.predict(X_test)
        RSS_rf = np.mean(pow((rf_test_pred- y_test),2))
        rf_scores.append(RSS_rf)

    mean_scores.append(np.mean(rf_scores)) # Update the mean scores
    std_scores.append(np.std(rf_scores)) # Update the standard deviation of scores

# Plot the results
plt.figure(figsize=(10, 6))
plt.errorbar(num_trees_list, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('Number of Trees')
plt.ylabel('R-squared (mean ± std)')
plt.title('Random Forest Regression Performance vs. Number of Trees')
plt.show()


# We will use X_raw created for this task previously, which contains the indicators from the original non-standardized data
inertia_scores = [] # List to store the inertia scores

for i in range(1, 11):  
    clustering_model = KMeans(n_clusters=i, random_state=110)
    clustering_model.fit(X_raw)
    inertia = clustering_model.inertia_ # Inertia scores are calculated
    inertia_scores.append(inertia)
    

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


kmeans_model = KMeans(n_clusters=4, random_state=110)
cluster_labels = kmeans_model.fit_predict(X_raw)
cluster_labels_df = pd.Series(cluster_labels, name='Cluster')
# The new dataset contains the original dataset and the added clusters
students_with_clusters = original_data.join(cluster_labels_df, how='outer')  

# Create histograms for each variable, separated by cluster
for i in range(students_with_clusters.shape[1] - 1):
    col_name = students_with_clusters.columns[i]

    # Set the figure size for each variable
    plt.figure(figsize=(10, 6))

    # Group by 'Cluster'
    cluster_groups = students_with_clusters.groupby('Cluster')

    # Create subplots for each cluster
    for cluster, data in cluster_groups:
        plt.hist(data[col_name], bins='auto', alpha=0.7, label=f'Cluster {cluster}')

    # Set labels and title for the entire plot
    plt.title(f'Histogram of {col_name} by Cluster')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot for each variable
    plt.show()


plt.figure(figsize=(15, 10))
plt.subplot(2,3,1)
sns.scatterplot(x='absences', y='goout', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Going out')

plt.subplot(2,3,2)
sns.scatterplot(x='absences', y='freetime', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Freetime')

plt.subplot(2,3,3)
sns.scatterplot(x='absences', y='final_grade', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Final Grade')

plt.subplot(2,3,4)
sns.scatterplot(x='goout', y='freetime', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Going out')
plt.ylabel('Freetime')

plt.subplot(2,3,5)
sns.scatterplot(x='goout', y='final_grade', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Going out')
plt.ylabel('Final Grade')

plt.subplot(2,3,6)
sns.scatterplot(x='freetime', y='final_grade', hue='Cluster', data=students_with_clusters, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Freetime')
plt.ylabel('Final Grade')

plt.show()

silhouette_scores = []
for i in range(2, 11):
    model = AgglomerativeClustering(n_clusters= i)
    cluster_labels = model.fit_predict(X_raw)
    silhouette_avg = silhouette_score(X_raw, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters={i}, the average silhouette_score is {silhouette_avg}")
    
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

optimal_n_clusters = 2  

# Apply Agglomerative Clustering
agglomerative_model = AgglomerativeClustering(n_clusters=optimal_n_clusters)
cluster_labels_2 = agglomerative_model.fit_predict(X_raw)
cluster_labels_df_2 = pd.Series(cluster_labels_2, name = 'Cluster')

# Add the cluster labels to the original DataFrame
students_with_clusters_2 = original_data.join(cluster_labels_df_2, how = 'outer')

# View the distribution of samples in each cluster
print(students_with_clusters_2['Cluster'].value_counts())

# Analyze the characteristics of each cluster
cluster_means = students_with_clusters_2.groupby('Cluster').mean()
print(cluster_means)


# Create histograms for each variable, separated by cluster
for i in range(students_with_clusters_2.shape[1] - 1):
    col_name = students_with_clusters_2.columns[i]

    # Set the figure size for each variable
    plt.figure(figsize=(10, 6))

    # Group by 'Cluster'
    cluster_groups = students_with_clusters_2.groupby('Cluster')

    # Create subplots for each cluster
    for cluster, data in cluster_groups:
        plt.hist(data[col_name], bins='auto', alpha=0.7, label=f'Cluster {cluster}')

    # Set labels and title for the entire plot
    plt.title(f'Histogram of {col_name} by Cluster')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.legend()

    # Show the plot for each variable
    plt.show()


plt.figure(figsize=(15, 10))
plt.subplot(2,3,1)
sns.scatterplot(x='absences', y='goout', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Going out')

plt.subplot(2,3,2)
sns.scatterplot(x='absences', y='freetime', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Freetime')

plt.subplot(2,3,3)
sns.scatterplot(x='absences', y='final_grade', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Absences')
plt.ylabel('Final Grade')

plt.subplot(2,3,4)
sns.scatterplot(x='goout', y='freetime', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Going out')
plt.ylabel('Freetime')

plt.subplot(2,3,5)
sns.scatterplot(x='goout', y='final_grade', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Going out')
plt.ylabel('Final Grade')

plt.subplot(2,3,6)
sns.scatterplot(x='freetime', y='final_grade', hue='Cluster', data=students_with_clusters_2, palette='viridis', s=50)
plt.title('Scatter Plot Colored by Clusters')
plt.xlabel('Freetime')
plt.ylabel('Final Grade')

plt.show()


