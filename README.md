# product_recommendation

Uses UK E-Commerce Dataset, containing 25,900 unique transactions, from 2010/2011

data/ contains processed versions of the original dataset.

models.py contains code for two recommendation models:
- **PopularityRecommender**: recommends the most popular "new" products (ie. not yet purchased by the target user) across all customers.
- **ColabFilteringRecommender**: uses collaborative filtering on customer-customer similarity matrix to recommend "new" products based on what customer similar to the target customer purchased.

evaluation.ipynb contains code to split the input data, train models and generate predictions, and compare the two models using F1 Score / Precision / Recall across a range of number of recommendations per user.

Comparing the F1 Score at 4 recommendations per user, the Collaborative Filtering Model shows a 123% improvement over the

*NB: evaluating recommender systems , which explains why the absolute F1 Score returned by the collaborative filtering model is low (0.0631), hence the focus on comparative evaluation*
