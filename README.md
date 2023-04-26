# Reddit NLP Classifier

---

## Overview & Problem Statement

Subreddit moderators face challenges with overlapping contents and users between two fashion-related subreddits, r/malefashionadvice and r/femalefashionadvice. 
To address this issue, we will leverage APIs and Natural Language Processing (NLP) techniquest to collect and analyze data from the two subreddits. The goal of this project is to develop a machine learning model that can accurately classify posts from each subreddit with a test accuracy of at least 0.8. By developing a model, moderators will be able to better manage their subreddits and provide more targeted content to their users. 

**What is Reddit?**: Reddit is a social news forum and discussion website that was founded in 2005. It serves as a platform where users can share their interests, hobbies, and passions. The content on Reddit is organized by subject into user-created communities, which are known as subreddits. Each subreddit has a specific focus and its own set of rules established by moderators. Users can interact with subreddits by posting comments, upvoting, or downvoting content. An upvote indicates that a post or comment is valuable and relevant to the subreddit or discussion, while a downvote indicates that it is irrelevant or unhelpful. These votes are used to determine the popularity of content, with the most popular posts and comments rising to the top of the subreddit page.

**General Reddit Stats**: According to [Semrush](https://www.semrush.com/blog/most-popular-social-media-platforms/), Reddit is the 17th most popular social media platform with 430 million monthly active users and 1.5 billion monthly visits to the site (published November 2022). Also, [Statista report](https://www.statista.com/study/72723/social-media-reddit-in-the-united-states-brand-report/?locale=en) reported that Reddit has a higher share of users in 18-29 years old, male, a high income, and live in cities and urban areas than other social networks (published in August 2022).  

--- 

## Data

The below two subreddits are used for this project. Their posts were collected using the [Pushshift's API](https://github.com/pushshift/api). 

- [r/malefashionadvice](https://api.pushshift.io/reddit/search/submission/?subreddit=malefashionadvice)
- [r/femalefashionadvice](https://api.pushshift.io/reddit/search/submission/?subreddit=femalefashionadvice)

--- 

## Exploratory Data Analysis 

After conducting an exploratory data analysis, we have identified several distinctive features that could prove useful in developing models. Our analysis has revealed unique characteristics such as language use, topic, and sentiment, among others. 

**The Word Count by Subreddit**: A new column called `word_count` was created to count the number of words in each subreddit post. The histogram charts represent the distribution of word counts by each subreddit. While both charts show right-skewed distribution, comments in `r/malefashionadvice` tend to have more a lower word count than those in `r/femalefashionadvice`.
![word_count_by_subreddit](/images/word_count.png)

**Counting A Single Word using CountVectorizer**: The below bar charts represent the most occuring single word in each subreddit. The most common single words in `r/malefashionadvice` include fashion items like `pant`, `shirt`, and `suits`, while `r/femalefashionadvice` had words like `dress`, `pants`, and `jean`. This suggests that there are differences in fashion interests between the two subreddits. 
![counting_words_1_word](/images/15_most_common_words_1.png)

**Counting Two-Word Combinations using CountVectorizer**: The below yellow chart shows top 15 two-word combinations occurring on both subreddits. These combinations include phrases like `look like`, `feel like`, `look good`, `don't know`, `don't think`, and `make look`, suggesting that users on both subreddits are seeking fashion advice. However, when looking at the top 15 two-word combinations on each subreddit separately, there are some differences. On `r/malefashionadvice`, the most frequently used phrases include `sport coat` and `slim fit`, while on `r/femalefashionadvice`, popular phrases include `high rise`, `straight leg`, `high waisted`, `wide leg`, and `skinny jeans`. This indicates that there are different fashion styles being discussed on each subreddit. Also, users on `r/malefashionadvice` often share brand names such as `Loro Piana`, which were frequently mentioned. 
![counting_words_2_words_both](/images/15_most_common_words_2_all.png)
![counting_words_2_words](/images/15_most_common_words_2.png)

These features provide valuable insights into the data and can be leveraged to create models that more accurately capture the patterns and trends present in the data. By incorporating these characteristics into our modeling process, we can improve our ability to make accurate predictions and generate actionable insights.

---

## Modeling

We used four different techniques, including Logistic Regression and Multinomial NB, along with Count Vectorizer and TF-IDF Vectorizer. The data was split into training and testing sets using an 80/20 split ratio. To ensure optimal performance, we utilized Grid-search Cross-validation with 5-folds to identify the best hyperparameters for each model. We then evaluated the performance of each model using various metrics such as train and test accuracy, and ROC AUC score. In addition, we used a confusion matrix to visually represent the performance of the models and review classification metrics such as precision and sensitivity (recall). By utilizing these techniques, we were able to develop models that accurately classified posts from both r/malefashionadvice and r/femalefashionadvice.

---

## Evaluation 

Here are scores and metrics to evaluate different classifier models and forms of vectorization.

|Transformer Type|Model Type|Train Accuracy|Test Accuracy|Precision|Recall|F1-Score|AUC|
|--|--|--|--|--|--|--|--|
|Count Vectorizer|Multinomial NB|0.933|0.810|0.824|0.787|0.805|0.901|
|Count Vectorizer|Logistic Regression|0.963|0.787|0.761|0.838|0.797|0.872|
|TF-IDF Vectorizer|Multinomial NB|0.952|0.817|0.841|0.782|0.810|0.906
|TF-IDF Vectorizer|Logistic Regression|0.880|0.795|0.777|0.829|0.802|0.870|

-- 

## Conclusion & Recommendations

Based on the evaluation metrics provided, both the TF-IDF Vectorizer with Multinomial Naive Bayes model and the Count Vectorizer with Multinomial Naive Bayes model achieve good accuracy in classifying posts from the two subreddits.

However, the TF-IDF Vectorizer with Multinomial Naive Bayes model achieves a slightly higher test accuracy of 0.817, indicating that this model may be better at accurately classifying posts from the two subreddits. On the other hand, the Count Vectorizer with Multinomial Naive Bayes model achieved higher recall values, which indicates that this model may be better at identifying all the posts from a particular subreddit.

The choice of vectorization technique depends on the specific needs and goals of the project. If the priority is to accurately classify posts from the two subreddits, then the TF-IDF Vectorizer with Multinomial Naive Bayes model may be the better choice. However, if the priority is to ensure that all posts from a particular subreddit are identified, then the Count Vectorizer with Multinomial Naive Bayes model may be more suitable.

In any case, it is recommended to conduct further analysis and fine-tuning of both models to improve the precision and recall values, particularly for the subreddit with lower recall. This could involve exploring different feature selection techniques or adjusting the hyperparameters of the models.

-- 

## Data Dictionary 

|Feature|Type|Dataset|Discription|
|----|----|----|----|
|subreddit|object|Reddit's two subreddits (`r/malefashionadvice` & `r/femalefashionadvice`)|Subreddit (Reddit's community) name|
|body|object|Reddit's two subreddits (`r/malefashionadvice` & `r/femalefashionadvice`)|Actual text from the post|
|created_utc|int64|Reddit's two subreddits (`r/malefashionadvice` & `r/femalefashionadvice`)|Date of submission creation(epoch time)|