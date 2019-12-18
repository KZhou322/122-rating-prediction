### Final_Project.ipynb
This notebook has inline comments to explain any changes that may be needed. Most of these are to change file paths when reading CSVs. It will produce predictions for one group of the data. This group can be changed to see how each group behaves and at the end of the notebook the user can choose to append the predictions to a final CSV or to output it by itself. 
All predicitons are in the all_predictions.csv file, and the predictions for just the best cluster are in predictions_group_1.csv

The code is separated into sections for Clustering, Building the Model, Predicting, and Outputting Results.
There are a number of warnings that come up while running the cells. These are all because of depracated methods with the use of tensorflow 2.0.

### cluster.py
This script should be run in a folder containing the data file 'user_history.csv' using the command 'python cluster.py'. It will cluster the website history data and generate two graphs, which the user can choose to save, and will automatically create the following files:
cluster_labels.npy
centroids.npy
ids.npy
where the cluster labels and ids files indicate which user ids belong to which cluster. The centroids file indicates the centroids of the three clusters generated.
