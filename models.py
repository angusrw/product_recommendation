import itertools
import time

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class Recommender:

    def __init__(self,name,maxrec):
        """
        name: name of recommender
        maxrec: maximum number of recommendations to store for each customer
        """
        self.name = name
        self.maxrec = maxrec
        self.df_recommendations = None

    def train(self):
        raise NotImplemented()

    def recommend(self,customers,numrec):
        """
        Return top n recommendations for given customers
        customers: list of customer ids to get recommendations for
        numrec: number of recommendatios to return for each customer
        """
        r = self.df_recommendations[self.df_recommendations['cid'].isin(customers)]
        r = r[r['rank']<=numrec]
        return r


class PopularityRecommender(Recommender):

    def train(self,data):
        """
        Create df containing recommendations for each customer based on most popular items in dataset that customer hasn't bought
        data: interaction matrix, containing customer's interaction (purchases) with each product
        """
        s=time.time()
        # get list of most popular products in dataset
        im_melt = pd.melt(data)
        prod_counts = im_melt[im_melt['value']>0].pid.value_counts()
        pop_prods = prod_counts.index

        # create df to put recommendations in (contains maxrec*rows for each customer)
        df_recom = pd.DataFrame(columns=['cid','pid','rank'])
        df_recom['cid'] = list(itertools.chain(*[[c]*self.maxrec for c in data.index]))
        df_recom['rank'] = list(itertools.chain(*[np.linspace(1,self.maxrec,self.maxrec).astype(int)]*len(data.index)))

        # for each customer
        for i,r in data.iterrows():
            # get list of 'new' products - ie. haven't been purchased before by customer
            new_prods = set(r[r==0].index)
            # remove items from rank that aren't 'new'
            filter_rank = [x for x in pop_prods if x in new_prods]
            # add filter_rank to recommendations df
            df_recom.loc[df_recom['cid']==i,'pid'] = filter_rank[:self.maxrec]

        self.df_recommendations = df_recom
        print(f"{self.name} training time: {round((time.time()-s)/60,2) }mins")


class ColabFilteringRecommender(Recommender):

    def train(self,data):
        """
        Create df containing recommendations for each customer based on collaborative filtering
        data: interaction matrix, containing customer's interaction (purchases) with each product
        """
        def predict_product_rating(cid,pid,similarity,k_neighbours=4):
            """
            Calculates recommendation rating for a product for a specific customer based on customers k nearest neigbours
            cid: id of customer
            pid: id of product
            similarity: user-user similarity matrix
            k_neighbours: number of neighbours to compute rating from
            """
            # get sorted list of most similar users that tried the product
            similar_prods = data[pid].loc[similarity.index]
            # get top k neighbours from sorted list of most similar users that tried product
            similar_prods_filled = similar_prods[similar_prods>0][:k_neighbours]

            # if customer in question is in list of similar customers, remove and add another similar customer
            if cid in similar_prods_filled.index:
                similar_prods_filled = similar_prods[similar_prods>0][1:k_neighbours+1]

            # get similarity scores and ratings
            similarity_list = similar_prods_filled.values
            rating_list = similarity.loc[similar_prods_filled.index].values

            #make weighted rating prediction based on similarity or return avg
            try:
                return sum(similarity_list*rating_list)/sum(similarity_list)
            except ZeroDivisionError:
                return rating_list.mean()


        s=time.time()

        # get user-user similarity matrix
        df_cos_sim = pd.DataFrame(index=data.index, columns=data.index, data=cosine_similarity(data))

        # create df to put recommendations in (contains maxrec*rows for each customer)
        df_recom = pd.DataFrame(columns=['cid','pid','rank'])
        df_recom['cid'] = list(itertools.chain(*[[c]*self.maxrec for c in data.index]))
        df_recom['rank'] = list(itertools.chain(*[np.linspace(1,self.maxrec,self.maxrec).astype(int)]*len(data.index)))

        #for each customer get ratings for products they haven't bought
        count = 0
        for i,r in data.iterrows():
            if count%400==0: print(f"{count}/{data.shape[0]}")
            count+=1
            # get sorted row of similar customers
            cust_sim_scores = df_cos_sim.loc[i].sort_values(ascending=False)
            # get list of 'new' products
            new_prods = set(r[r==0].index)
            dict_prod_ratings = dict()
            # for each new product get predicted rating
            for j,prod in enumerate(new_prods):
                dict_prod_ratings[prod] = predict_product_rating(i, prod, similarity=cust_sim_scores)

            # sort prods by rating, add to recommended df
            df_recom.loc[df_recom['cid']==i,'pid'] = sorted(
                dict_prod_ratings, key=dict_prod_ratings.get, reverse=True)[:self.maxrec]

        self.df_recommendations = df_recom
        print(f"{self.name} training time: {round((time.time()-s)/60,2) }mins")
