# This article has been depreciated. For an update version, check this [blogpost](https://shuaiw.github.io/2016/07/19/data-science-project-workflow.html).

By [Shuai Wang](https://github.com/ShuaiW)

### Prologue: Use the Right Tool

"If the only tool you have is a hammer, you tend to see every problem as a nail." -- Abraham Maslow

According to a [talk](https://www.import.io/post/how-to-win-a-kaggle-competition/) by Anthony Goldbloom, CEO of Kaggle, there are only two winning approaches:

0. Handcrafted 
0. Neural Networks

In the first category, it "has almost always been ensembles of decision trees that have won". [Random Forest](https://en.wikipedia.org/wiki/Random_forest) used to be the big winner, but [XGBoost](https://github.com/dmlc/xgboost) has cropped up, winning practically every competition in the structured data category recently.

On the other hand, for any dataset that contains images or speech problems, deep learning is the way to go. And instead of spending almost none of their time doing feature engineering, winners spend their time constructing [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network). 

So, the major takeaway from the talk is that **if you have lots of structured data, the handcrafted approach is your best bet, and if you have unusual or unstructured data your efforts are best spent on neural networks**.

I've recently conducted a [use case proposal](https://docs.google.com/document/d/1YrcN60jrYJwDL5F8yyd6_Yj279GSZMNBEOHwTBxwP8c/edit?usp=sharing) of a kaggle competition, and I analyzed (in the appendix) why deep nerual networks, while theoratically being able to approximate any functions, in practice are easliy dwarfed by ensemble tree algorithms in terms of efforts put in vs. results. That said, stacking deep nets in the final ensemble might boost performance, as long as the features extracted are not highly correlated with those from ensemble trees.

======

### Workflow
0. [Divide Forces](#divide-forces)
0. [Data Setup](#data-setup)
0. [Literature Review](#literature-review)
0. [Establish an Evaluation Framework](#establish-an-evaluation-framework)
0. [Exploratory Data Analysis](#exploratory-data-analysis)
0. [Preprocessing](#preprocessing)
0. [Feature Engineering](#feature-engineering)
0. [Model(s) Tuning](#models-tuning)
0. [Ensemble](#ensemble)

======


### Divide Forces

1 + 1 is not always equal to 2 when it comes to team collaboration. More often, it is smaller than 2 due to factors such as individual favorite scripting langauge, workflow, etc. Besides using collaborative tools (Skype, svn, git) to keep effective communication, how to divide force effciently is key to get great results as a team.

According to Kaggle winners' interviews, forming teams can make them more competitive as "combining features really helps improve performance". At the same time, it helps to specialize, having different members focus on different stages of the data pipeline. 

So one plausible plan is as follows:

* Complete data setup
* Review literature
* Establish a evaluation framework as a team, either single model nested CV or ensemble
* People with strong statistical background should tackle the EDA task
* People with strong scripting (such as Python) skills should then, with insights from the EDA phase, preprocess the data (this step might involve SQL depending on data sources)
* People with domain expertise, strong statistical skills, and/or practical experience in working with features should focus on the feature engineering part (**Note**: for deep learning project this step is not always necessary). The output of feature engineering, as suggest by the [1st place Home Depot Product Search Relevance](http://blog.kaggle.com/2016/05/18/home-depot-product-search-relevance-winners-interview-1st-place-alex-andreas-nurlan/), should be serialized pandas dataframes.
* People with machine learning expertise should handle model tuning and ensmeble parts

So in traditional machine learning problem, after the preprocessing step, the people who work on feature engineering should be producing new feature sets frequently so that machine learning people can tune and train models continuously. On the other hand, in deep learning regime most of the time would be spent on net architecture design and model/parameters tuning.

**Clearly documented, reproducible** codes and data files are crucial.

======

### Data Setup

* **Get the data**: usually for competition such as [Kaggle](https://www.kaggle.com/) we can get the data effortlessly by downloading from its website. If we are working with clients, getting data might require further efforts, such as accessing their database or cloud storage.
* **Store the data**: Either locally or on the cloud, depending on the situation. It's beneficial to have some backup mechanism in place.

======

### Literature Review

This may be one of the most important (yet overlooked) part for competing in Kaggle, or doing data science in general. To avoid reinventing the wheels and get inspired on how to preprocess, engineer and model the data, it's worth spend 1/10 to 1/5 of the project time just researching how people previously dealt with similar problems/datasets. Some good places to start are:

* [No Free Hunch](http://blog.kaggle.com/): the official Kaggle blog, artciles under category **Tutorials** and **Winner's Interviews**
* [Kaggle](https://www.kaggle.com/competitions): Browse through active/completed competitions, and study the similar projects to the one we are working on
* Google is our friend :)

Time spent on literature review is time well spent.

======

### Establish an Evaluation Framework

Having a sound evaluation framework is crucial for all the works that follow: if we use suboptimal metrics or don't have an effective cross validaton strategy that could gudie us tune generalizable models, we are aiming at the wrong target and wasting our time. 

**Metrics**

In terms of the metrics to use, if we are competing on Kaggle we can find it under Evalution. If, however, we are working on some real-world data problem where we are free to craft/choose the ruler to measure how good (or bad) our models are, cares should be given about choosing the 'right' metric that makes the most sense for a domain the problem/data at hand. For example, we all know wrongfully classifying a spam email as non-spam does less harm than classifying the non-spam to be spam (Imagine missing an important meeting email from your boss...). Similar case for medical diagnosis. In such situations, simple metric such as accuracy is not enough and we might want to consider other metrics, such as precision, recall, F1, ROC AUC score, etc.

A good place to view different metrics and their use cases is [sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) page.


**Cross Validation (CV) Strategies**

The key point for machine learning is to build models that can generalize on unseen data, and a good cross validation strategy help us achieve that (while a bad strategy misleads us and gives us false confidence). 

Some good read on cross validation stategies:

* [10-fold CV](https://www.kaggle.com/c/titanic/forums/t/3244/cross-validation-methods): some discussion regarding why 10 is the magic number
* [Nested cross-validation](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1397873/pdf/1471-2105-7-91.pdf): Varma and Simon concluded that the true error of the estimate is almost unbiased relative to the test set when nested cross-validation is used
* [Cross validaton strategy when blending/stacking](https://www.kaggle.com/forums/f/15/kaggle-forum/t/18793/cross-validation-strategy-when-blending-stacking) 

Here is one simple example of nested cross-validation in Python using scikit-learn:

``` python 
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X_train = ... # your training features
y_train = ... # your training labels

gs = GridSearchCV(
  estimator = RandomForestClassifier(random_state=0),
  param_grid = {
    'n_estimators': [100, 200, 400, 600, 800],
     # other params to tune
     }
  scoring = 'accuracy',
  cv = 5
)

scores = cross_val_score(
  gs,
  X_train,
  y_train,
  scoring = 'accuracy',
  cv = 2
)

print 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
```

In a nutshell, the inner loop `GridSearchCV` is used to tune parameters, while the outer loop `cross_val_score` is used to train with optimal parameters and report an unbiased score. This is known as **2x5 cross-validation** (although we can do 3x5, 5x4, or any fold combinations that work for our data size and computational constraints).

======

### Exploratory Data Analysis (EDA)

EDA is an approach to analyze data sets to summarize their main characteristics, often with plots. EDA helps data scientists get a better understanding of the dataset at hands, and guide them to preprocess data and engineer features effectively. Some good resources to help carry out effective EDA are:

* [Think Stats 2](http://greenteapress.com/thinkstats2/): an introduction to probability and statistics for Python programmers (its github repository holds some useful codes and ipython notebooks for conducting exploratory data analysis)
* [Data Science with Python & R: Exploratory Data Analysis](https://www.codementor.io/python/tutorial/data-science-python-r-exploratory-data-analysis-visualization)
* There are people who usually post their code for EDA on Kaggle forum at the beginning of each competition, such as [this](https://www.kaggle.com/cast42/santander-customer-satisfaction/exploring-features). So check out the forum frequently.

======

### Preprocessing

The one and only reason we need to preprocess data is so that a machine learning algorithm can learn most effectively from them. Specifically, three issues need to be addressed:

* Sources: we need to integrate data from multiple sources to improve predictability 
* Quality: incomplete, noisy, inconsistent data
* Format: incompatible data fortmat. E.g., generalized linear model requires one-hot encoding of ordinal variables. Another case in point is to use word embedding (vector) rather than text as input for nlp algorithms.

In practice, tasks in data preprocessing include:

* **Data cleaning**: fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies
* **Data integration**: use multiple data sources and join/merge data
* **Data transformation**: normalize, aggregate, and embed (word)
* **Data reduction**: reduce the volume but produce the same or similar analytical results (e.g., PCA)
* **Data discretization**: replace numerical attributes with nominal/discrete ones

Also check [CCSU course page](http://www.cs.ccsu.edu/~markov/ccsu_courses/datamining-3.html) and [scikit-learn documentation](http://scikit-learn.org/stable/modules/preprocessing.html) for data preprocessing in general and how to do it effectively in Python.

======

### Feature Engineering

"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." -- Andrew Ng

While in deep learning we usually just normalize the data (such that the image pixels have zero mean and unit variance), in traditional machine learning we need handcrafted features to build accurate models. Doing feature engineering is both art and science, and requires iterative experiments and domain knowledge. Feature engineering boils down to feature selection and creation.

**Selection**

scikit-learn offers some great feature selection [methods](http://scikit-learn.org/stable/modules/feature_selection.html). They can be categoried as one of the following:

* Removing features with low variance
* Univariate feature selection
* Recursive feature elimination
* Selecting from model

Here is one simple example of selecting features from model in Python using scikit-learn:

``` python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

X_train = ... # your training features
y_train = ... # your training labels

# can be any estimator that has attribute 'feature_importances_' or 'coef_'
model = RandomForestClassifier(random_state=0) 

model.fit(X_train, y_train)

fs = SelectFromModel(model, prefit=True)

X_train_new = fs.transform(X_train) # columns selected
```

In the above example, features with zero importance (feature_importances_ = 0) will be eliminated. 

Some good articles about feature selection are:

* [How do I perform feature selection](https://www.quora.com/How-do-I-perform-feature-selection)
* [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](http://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
* [Streamlining feature engineering](http://radar.oreilly.com/2014/06/streamlining-feature-engineering.html)

**Creation**

This is the part where domain knowledge and creativity come in. For example, insurers can create features that help their machine learning model better identify customers with higher risks; similar for geologists who work with geological data.

But here are some general methods that help you create features to boost model performance:

* Add zero_count for each row (especially for sparse matrix)
* Seperate date into year, month, day, weekday or weekend, etc.
* Add percentile change from feature to feature (or other interactions among features)

After adding new handcrafted features, we need to perform another round of feature selection (e.g., using `SelectFromModel` to elimiate non-contributing features). Note that different classifiers might select different features, and it's imoprtant that features selected using a certain classifier are later trained with the **same** classifier. For classifiers that don't have either `feature_importances_` or `coef_` attribute (e.g., nonparametric classifiers such as [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)), the best way is to cross validate the features selected from various classifiers (i.e., to select the set of feature that has the highest CV score).

Feature engineering is the jewel in crown of machine learning. As machine learning professor [Pedro Domingos] (http://homes.cs.washington.edu/~pedrod/) puts it: "...some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used."

======

### Model(s) Tuning

If we've got this far, model tuning is actually the easy part: we simply need to provide the parameter search space (coarse to refined) with certain classifiers, and hit the run button and let the machines do all the heavy lifting. In this regard, the nest CV strategy discussed above will do the job perfectly.

If only things were this simple. In practice, however, we face a big constraint: time (deadline to deliver the model to clients or Kaggle competition). To add more complexity, single model never generates the best results, meaning we need to stack many models together to deliver great performance. How to effectively stack (or ensemble) models will be discussed in the next section. 

So, one alternative is to optimize the search space automatically, rather than manually setting each parameter from the coarse to the refined. Several Kaggle winners use and recommend [hyperopt](https://github.com/hyperopt/hyperopt), a Python library for serial and parallel parameter optimization. 

More notes/codes will be added w.r.t how to effectively use hyperopt. 

======

### Ensemble

Training one machine learning model (e.g., XGBoost, Random Forest, KNN, or SVM) can give us decent results, but not the best ones. If in practice we want to boost the performance, we might want to [stack](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) models using a combiner, such as [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). This [1st place solution](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov) for [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge) is AWESOME. Here is one way how we do stacking:

* Preprocessing and feature engineering (which are addressed above)
* First layer: Use nested CV to train N models (1st place of Otto competition used same fold indices), generating N * C (C== num_class) probabilistic predictions (or, meta features) for each data point.
* Second layer
  - Single combiner: use the N * C meta features from first layer, plus optionally handcrafted features to train another model (e.g, logistic regression) with nested CV.
  - Multiple combiners: train multiple single combiner with nest CV (1st place of Otto competition used random indices across different combiners).
* Third layer (for multiple combiners): cross validate the weights of combiners in the second layer (can be simple linear weights or complex ones such as geometric mean)


