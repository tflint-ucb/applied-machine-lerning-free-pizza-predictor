# RedditRandomActsofPizza_Predictor
Predict whether a pizza request will be granted through the Random Acts of Pizza community on Reddit

## Background
This analysis uses the Random Acts of Pizza dataset (https://www.kaggle.com/c/random-acts-of-pizza/data) generated from Reddit. This subreddit allows anyone to submit a request to ask for a free pizza, and readers may respond to specific posts and send a free pizza.

Our goal is to produce a model to predict the likelihood of receiving a pizza and to deliver insights into what we think makes for a great free pizza request. This binary classification problem involves free-form textual content and attached metadata about the post and Reddit community members.

In Part 1 of this analysis, we explore the the data, and generate visualizations and summary statistics to reveal patterns between requests that did receive pizza and those which do not. We will also run preliminary sentiment analysis on the request text and explored additional features that we can derive from the dataset.

The breakdown of positive (received pizza) to negative (did not receive pizza) cases of receiving a pizza in the training data set is about 25% positive and 75% negative cases. Since the data is unbalanced, we will perform resampling/balancing techniques. We will use accuracy scores, precision, recall, and f1 scores to measure performance.

To begin our exploration of baseline models, we will implement Logistic Regression and Bernoulli Naive Bayes' models using a bag of words approach to vectorize the text data and some data balancing techniques. We use the Logistic Regression model on textual data and in-built balancing as our main Baseline model because it performs the best among the other initial models.

Learning from the explanatory data analysis and results of the baseline models from Part 1, we further cleaned the data and redesigned our features, such as scaling to normalize our newly cleaned numeric and feature engineering, in Part 2. We then use this enhanced dataset to run additional variations of predictive models, such as Logistic Regression, K-Nearest Neighbors, Random Forest, Transformers, and XGBoost.

We have provided a summary of measure performance at the end to show which models fit best on the cross-validated training set. We then picked the two best models and ran them on the test data once.

Lessons from this exercise could be valuable to not only the Reddit community, but other social platforms that wish to improve their user engagement through NLP techniques.

## Outline
1. Baseline Submission
	- 1.1. Introduction
	- 1.2. EDA on textual data
	- 1.3. EDA on non-textual data
	- 1.4. Baseline Models with both Numeric and Textual Features
        - Logistic Regression
            - Logistic Regression model that uses the in-built balancing function.
			- Logistic Regression model that does NOT use the in-built balancing function
			- Logistic Regression model that uses SMOTE balancing technique
		- Bernoulli Naive Bayes
			- Bernoulli Naive Bayes model on unbalanced data
			- Bernoulli Naive Bayes model on SMOTE balanced data
	- 1.5. Baseline Models with only Textual Features
		- Logistic Regression
			- Logistic Regression model that uses the in-built balancing function
			- Logistic Regression model with no balancing function
		- Bernoulli Naive Bayes
			- Bernoulli Naive Bayes model that on unbalanced data
	- 1.6. Summary
    
    
2. Final Submission	 
	- 2.1. Further data cleaning 
	- 2.2. Models on Numeric Features
		- Logistic Regression 
        - KNN
	- 2.3. More feature engineering
	- 2.4. Logistic Regression and KNN Enhancements
	- 2.5. Random Forest
	- 2.6. XGBoost
    - 2.7. BERT
    - 2.8. Test Performance
	- 2.9. Summary
   

Appendix



## Data Format
Aside from the requesters' text and title requesting for the pizza, there are some meta data like the number of comments the requesters has posted on the different subreddits, time stamp information of the request, number of upvotes...etc.

# Getting Started
Notebook W207_Final_Project.ipynb contains the eda and the different models we have tried. The final best performing model, xgboost, has the following performance on the test data: : accuracy = 0.8527, precision = 0.7277, recall = 0.6748, f1 = 0.7003.


# Prerequisites
Pip dependencies in the requirements.txt file.
* pip3 install -r requirements.txt 

Conda dependencies 
* conda install --file conda_requirements.txt

# Final Summary
In conclusion, we find that XGBoost and L2 regularized Logistic Regression were the top two models on the training data, but the XGBoost performs better on the test set (F1 score of 0.7003). XGBoost can handle high dimensionality and missing data very well can can run very quickly, giving it the advantage over Logistic Regression. XGBoost improves the shortcomings of Random Forests, such as better balancing techqniues. K-Nearest Neighbors underperforms Logistic Regression in most cases, and this could be due to the high dimensionality of our features. BERT model did not work that well, and it is likely to its heavier focus on the text data, which had a fair bit of noise.

To arrive at this conclusion, we explored the data and implemented simple models as our baseline in Part 1.

The baseline model exploration helped us to spot a couple of issues that need to be resolved, such as imbalanced data and overfitting. From our EDA, we see about only 25% of requests were granted pizzas. In an initial attempt to address the imbalanced data, we implemented SMOTE balancing and in-built balancing for the successful pizza requests. We also had a large number of features, which caused the curse of dimensionality problem. This also caused our models to heavily overfit in Part 1. In Part 2, we aim to address these challenges in detail. Thus, we implemented each enhancement separately to see its impact on model performance in Part 2. We also experimented with running models on only numeric, only textual, and both numeric and textual features.

Enhancements include:

- Balance the data
    - SMOTE
    - in-built balancing
- Clean and impute the outliers and missing values 
- Cross validation
- Text preprocessing and lemmatization
- Feature Engineering 
    - Adding features on punctuation count and sentiment score from text data
    - Adding features on time of day and day of week from some numeric features
- Scale the data
- Remove highly correlated numeric features
- Measure F1 score instead of accuracy
- Reduce the vocab size
    - Using only the top 1000 most frequent words 
    - Using L1 regularization 
    

We use F1 score, instead of accuracy, as a measure of performance because our original datset was heavily imbalanced.

We found that 'at retrieval' features, what happens on Reddit after the post was made, are more important than the 'at request' features, what happens right as the post is made. So, prolonged Reddit activity affects pizza request success rate more. We also discovered that word features, from request text, hold less predictive power than numeric features, request metrics and metadata. This means that request success is more determined by objective features like number of upvotes, rather than what words and sentiment a requestor uses in the post.

In summary, post popularity, measured via comments, and user activity, measured via user posts so far, likely increase the chance that a request receives a free pizza. Using words related to food insecurity and specific needs also increase the chance of receving a pizza.

From this analysis, we learned that:

Understanding the data is crucial in deciding on existing menu of prediction models that offer varying degrees of performance depending on data characteristics and issues. This includes understanding data anomalies or unexpected inputs and how different models behave with these.
This data understanding then guides feature engineering and selection, and hyperparameter tuning.
Baseline estimates, and using appropriate performance metrics, are important to gauge improvements in model outcomes.
Collaboration and leveraging on each others strengths can lead to a more efficient work flow in a team.
We aim for this analysis to provide helpful insight into user language patterns and behavior on not only Reddit, but also other online forums and social media platforms. Such work will aid in improving user engagement and retention.


# Future work 
1) Feature engineer or work on collecting some meta data that may offer more predictive power. Some ideas include: 
* which store location gave out the free pizzas in the past
* what keywords caught owner’s eyes
* which sentence the owner agree or disagree with the most (kinda similar to 2)
* which employee approved the request / brought the requests to the decision-maker 
* sales of the day //poor-performing stores probably can't give much 
* leftover or excess pizzas at end of day


2) Hyperparameter tune the xgboost model with HypOpt, which is a more efficient and "smart" tuning than GridSearchCV.

# Authors
Eric (Yue) Ling, Krutika Ingale, Mitch Abdon, Tanya Flint

# Acknowledgments

Natalie Ahn for giving tips about training the models, different techniques and recommendations for working with BERT transformers and transfer learning (concatenating numeric and BERT outputs). 

