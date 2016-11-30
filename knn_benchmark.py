import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


model = KNeighborsClassifier(n_neighbors=3)

examples = np.loadtxt('music_data_2class.csv', delimiter=',')

labels = np.loadtxt('music_labels_2class.csv')

model.fit(examples, labels)

num_folds = 10

kf = KFold(n_splits= num_folds, shuffle=True)

avg_performance = 0.0


for train, test in kf.split(examples, labels):
    batch_xs = examples[train]
    batch_ys = labels[train]

    test_xs = examples[test]
    test_ys = labels[test]

    preds = model.predict(test_xs)

    performance = accuracy_score(labels[test],preds)

    avg_performance += performance

    print ('Accuracy: ' + str(performance))


avg_performance = float(avg_performance) / float(num_folds)

print ("Average Accuracy: " + str(avg_performance))



