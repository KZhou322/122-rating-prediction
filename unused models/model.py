import numpy as np
import csv
import pandas as pd
import math
from random import randint
from collections import defaultdict
import sklearn.cluster as cluster

def load_rating_data(file):
    csv_reader = csv.DictReader(file)
    users = []
    products = []
    ratings = []
    ids = []

    # used to map user id to the index in the range 0 to n_users
    user_index = {}
    u = 0

    product_index = {}
    p = 0

    # only minor changes from provided code, puts data into np arrays
    for row in csv_reader:
        user = int(row["USER ID"])
        product = row["PRODUCT"]
        if not user in user_index:
            user_index[user] = u
            u += 1
        if not product in product_index:
            product_index[product] = p
            p += 1
        # We subtract 1 from the ids to get (0-based) indices
        users.append(user_index[user])
        products.append(product_index[product])
        ratings.append(float(row["RATING"]))
        ids.append(user)

    data = np.transpose(np.array([users, products, ratings, ids]))
    return data, user_index, product_index



def initialize(n_users, n_products, k):
    """Initalize a random model, and normalize it so that it has sensible mean and variance"""
    # (The normalization helps make sure we start out at a reasonable parameter scale, which speeds up training)
    user_features = np.random.normal(size=(n_users, k))
    product_features = np.random.normal(size=(n_products, k))
    raw_predictions = predict((user_features, product_features))
    
    s = np.sqrt(2*raw_predictions.std()) # We want to start out with roughly unit variance
    b = np.sqrt((5.5 - raw_predictions.mean()/s)/k) #We want to start out with average rating 5.5
    user_features /= s
    user_features += b
    product_features /= s
    product_features += b
    
    return (user_features, product_features)

def predict(model):
    """The model's predictions for all user/movie pairs"""
    user_features, product_features = model
    return user_features @ product_features.T

def single_example_step(model, user, product, rating):
    """Update the model using the gradient at a single training example"""
    user_features, product_features = model
    residual = np.dot(user_features[int(user)], product_features[int(product)]) - rating
    grad_users = 2 * residual * product_features[product] # the gradient for the user_features matrix
    grad_products = 2 * residual * user_features[user] # the gradient for the movie_features matrix
    user_features[user] -= learning_rate*grad_users
    product_features[product] -= learning_rate*grad_products

def train_sgd(model, epochs):
    """Train the model for a number of epochs via SGD (batch size=1)"""
    user_features, product_features = model
    # It's good practice to shuffle your data before doing batch gradient descent,
    # so that each mini-batch peforms like a random sample from the dataset
    shuffle = np.random.permutation(m) 
    shuffled_users = train_users[shuffle]
    shuffled_movies = train_products[shuffle]
    shuffled_ratings = train_ratings[shuffle]
    for epoch in range(epochs):
        for user, movie, rating in zip(shuffled_users, shuffled_movies, shuffled_ratings):
            # update the model using the gradient at a single example
            single_example_step(model, user, movie, rating)
        # after each Epoch, we'll evaluate our model
        predicted = predict(model)
        train_loss = np.mean((train_ratings - predicted[train_users, train_products])**2)
        test_loss = np.mean((test_ratings - predicted[test_users, test_products])**2)
        print("Loss after epoch #{} is: train/{} --- test/{}".format(epoch+1, train_loss, test_loss))




def all_examples_step(model):
    """Update the model using the gradient averaged over all training examples"""
    user_features, product_features = model
    # To average the gradient over all training examples, it's convenient to
    #    initialize arrays of zeros to hold the full gradients, and then update
    #    these arrays at each training example, just like in the SGD procedure
    grad_users = np.zeros(np.shape(user_features))
    grad_products = np.zeros(np.shape(product_features))
    # We only need to compute the model's predicted ratings once
    predicted = predict(model)
    for user, product, rating in zip(train_users, train_products, train_ratings):
        # Mimic the SGD procedure, but store the gradients so they can be averaged
        residual = predicted[user, product] - rating
        grad_users[user] += 2 * residual * product_features[product]
        grad_products[product] += 2 * residual * user_features[user]
    user_features -= learning_rate/m * grad_users # Update using the averaged gradients
    product_features -= learning_rate/m * grad_products

    
def train_full(model, epochs):
    """Train the model for a number of epochs using gradients estimated from the entire training set"""
    user_features, product_features = model
    for epoch in range(epochs):
        all_examples_step(model)
        predicted = predict(model)
        train_loss = np.mean((train_ratings - predicted[train_users, train_products])**2)
        test_loss = np.mean((test_ratings - predicted[test_users, test_products])**2)
        print("Loss after epoch #{} is: train/{} --- test/{}".format(epoch+1, train_loss, test_loss))



# number of clusters
k=3
# load website history data
raw_history_data = pd.read_csv("user_history.csv")
history_data = raw_history_data.to_numpy()

# cluster absed on website history
clusters = cluster.KMeans(n_clusters=k, n_init=10, max_iter=100)
cluster_labels = clusters.fit_predict(history_data[:,1:])

# divide ids into sets
id_sets = [set() for _ in range(k)]
for i in range(np.size(history_data, axis=0)):
    id_sets[cluster_labels[i]].add(history_data[i,0])



with open("user_ratings.csv") as train_file:
    data, user_index, product_index = load_rating_data(train_file)

# separate data into a list of numpy arrays, based on which cluster the user belongs to
clustered_data = [[data[i,:] for i in range(np.size(data, axis=0)) if data[i,3] in id_sets[j]] for j in range(k)]
clustered_data = [np.array(cluster) for cluster in clustered_data]

test_size = 0.1 # fraction of the data to use as test data
learning_rate = 0.005
k = 5 # the number of features (for each user/movie)

# run the model on each cluster
for cluster in clustered_data:
    n = np.size(cluster, axis=0)
    cutoff = int(n*test_size)
    # shuffle = np.random.permutation(n) 
    # shuffled_cluster = cluster[shuffle]
    test = cluster[:cutoff]
    train = cluster[cutoff:]

    test_users = test[:,0]
    test_products = test[:,1]
    test_ratings = test[:,2]
    test_ids = test[:,3]
    train_users = train[:,0]
    train_products = train[:,1]
    train_ratings = train[:,2]
    train_ids = train[:,3]

    m = len(train_ratings) # the size of the training set

    # how many users, products are in this subset
    n_users = np.size(np.unique(cluster[:,0], axis=0), axis=0) 
    n_products = np.size(np.unique(cluster[:,1], axis=0), axis=0)

    # print(n_users)
    # print(n_products)
    
    # here is where I stopped implementing the model, reasons described in writeup
    sgd_model = initialize(n_users, n_products, k)
    train_sgd(sgd_model, 10)


# full_model = initialize(n_users, n_products, k)
# learning_rate = 8. # Since we are averaging very sparse gradients,
# # the gradients will be small and we can use a large learning rate
# train_full(full_model, 100) # We only get a single update to the model from each epoch, so we'll need a lot more epochs