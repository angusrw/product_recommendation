# product_recommendation

Uses (UK E-Commerce Dataset)[https://www.kaggle.com/carrie1/ecommerce-data], containing 25,900 unique transactions, from 2010/2011

(data/)[https://github.com/angusrw/product_recommendation/tree/main/data] contains processed versions of the original dataset and saved test/train splits and recommendations from models.

(models.py)[https://github.com/angusrw/product_recommendation/blob/main/models.py] contains code for two recommendation models:
- **PopularityRecommender**: recommends the most popular "new" products (ie. not yet purchased by the target user) across all customers.
- **ColabFilteringRecommender**: uses collaborative filtering on customer-customer similarity matrix to recommend "new" products based on what customers similar to the target customer purchased.

(eval.py)[https://github.com/angusrw/product_recommendation/blob/main/eval.py] contains code for generating the evaluation metrics.

(train_eval.ipynb)[https://github.com/angusrw/product_recommendation/blob/main/train_eval.ipynb] contains code to split the input data, train models and generate predictions, and compare the two models using F1 Score / Precision / Recall across a range of number of recommendations per user.

Comparing the F1 Score at 4 recommendations per user, the Collaborative Filtering Model shows an improvement of 123% over the baseline model (this figure will differ between runs).

*NB: evaluating recommender systems can be difficult due to the nature of the task as recommending a list of predictions of some length, and then comparing that list to another list of purchases which might a different length, which explains why the absolute F1 Score returned by the collaborative filtering model is low (>0.1), hence the focus on comparative evaluation*
