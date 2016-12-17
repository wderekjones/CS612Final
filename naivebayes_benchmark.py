import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from confusionplot import plot_confusion_matrix


max_iters = 100


model = BernoulliNB()

examples = np.loadtxt('music_data_2class.csv', delimiter=',')

labels = np.loadtxt('music_labels_2class.csv')




x_train,x_test,y_train,y_test = train_test_split(examples,labels,test_size = 0.2, random_state=42)

num_folds = 5

kf = KFold(n_splits=num_folds, shuffle=True)


mean_performance = 0.0


for i in range(max_iters):

    avg_performance = 0.0

    for train, test in kf.split(x_train, y_train):
        batch_xs = x_train[train]
        batch_ys = y_train[train]

        test_xs = x_train[test]
        test_ys = y_train[test]

        model.fit(batch_xs,batch_ys)

        preds = model.predict(test_xs)

        performance = accuracy_score(test_ys, preds)

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


