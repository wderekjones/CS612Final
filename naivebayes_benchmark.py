import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from confusionplot import plot_confusion_matrix


max_iters = 100


model = GaussianNB()

examples = np.loadtxt('music_data_2class.csv', delimiter=',')

labels = np.loadtxt('music_labels_2class.csv')

model.fit(examples, labels)

num_folds = 5

kf = KFold(n_splits= num_folds, shuffle=True)


mean_performance = 0.0


for i in range(max_iters):

    avg_performance = 0.0

    for train, test in kf.split(examples, labels):
        batch_xs = examples[train]
        batch_ys = labels[train]

        test_xs = examples[test]
        test_ys = labels[test]

        preds = model.predict(test_xs)

        performance = accuracy_score(labels[test],preds)

        avg_performance += performance

    avg_performance = float(avg_performance) / float(num_folds)

    mean_performance += avg_performance


    print ("Average Accuracy" + " at step "+str(i)+": " + str(avg_performance))


mean_performance = float(mean_performance) / float(max_iters)



avg_misclass_error = 1 - mean_performance

print ("Misclassification Error =  " + str(avg_misclass_error))

finalpreds = model.predict(examples)

confusion = confusion_matrix(labels,finalpreds)

class_labels = [0,1,2,3,4,5,6,7,8]

plot_confusion_matrix(confusion,classes=class_labels,plotname='NaiveBayes_benchmark.png',normalize=True)


