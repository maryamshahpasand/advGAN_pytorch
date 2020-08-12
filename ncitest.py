import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import scipy.sparse
# from sklearn.feature_extraction.text import CountVectorizer as TC
# from sklearn.model_selection import train_test_split
# import random
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import LinearSVC
# import time
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import metrics
# from sklearn.feature_selection import SelectFromModel


Drebin_important_features= np.load('C:/Users/45028583/Desktop/Drebin_important_features.npy')
for i in range(len(Drebin_important_features)):
    if Drebin_important_features[i][2] in ['api_call', 'call']:
        print(Drebin_important_features[i])














FEATURES_SET = [    "feature",
    "permission",
    "activity",
    "service_receiver",
    "provider",
    "service",
    "intent",
    "api_call",
    "real_permission",
    "call",
    "url"]

mypath = "/short/yq94/feature_selection/data/drebin/feature_vectors/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print ("Reading csv file for ground truth ...")
ground_truth = np.loadtxt("/short/yq94/feature_selection/data/drebin/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)

print ("Reading positive and negative texts ...")
pos = []
neg = []
for virus in onlyfiles:
    if virus in ground_truth[:, 0]:
        #if len(pos) < 10:
        pos.append(mypath+virus)
    else:
        #if len(neg) < 10:
        neg.append(mypath+virus)

print("Loading Malware and Goodware Sample Data")
AllSampleNames = pos + neg
print("Loaded samples")
TestSize =0.3


FeatureCountVectorizer = TC(input='filename',encoding ='utf-8', tokenizer=lambda x: x.split('\n'), token_pattern=None,\
                            binary=False)
x_ = FeatureCountVectorizer.fit_transform(pos + neg)
features=FeatureCountVectorizer.get_feature_names()
print("FeatureCountVectorizer shape: %d", x_.shape)

# label malware as 1 and goodware as -1
Mal_labels = np.ones(len(pos))
Good_labels = np.empty(len(neg))
Good_labels.fill(0)
y = np.concatenate((Mal_labels, Good_labels), axis=0)


x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=TestSize,
                                                                            random_state=random.randint(0, 100))
print("Test set split = %s", TestSize)
print("train-test split done")


#
# step 3: train the model
print("Perform Classification with SVM Model")
Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

T0 = time.time()
# if not Model:
Clf = GridSearchCV(LinearSVC(), Parameters, cv=5, scoring='f1', n_jobs=-1)
SVMModels = Clf.fit(x_train, y_train)
print(
    "Processing time to train and find best model with GridSearchCV is %s sec." % (round(time.time() - T0, 2)))
BestModel = SVMModels.best_estimator_
print("Best Model Selected : {}".format(BestModel))
print("The training time for random split classification is %s sec." % (round(time.time() - T0, 2)))
print("Enter a filename to save the model:")
# filename = raw_input()
filename = "SVM"
# dump(Clf, filename + ".pkl")
# else:
#     SVMModels = load(Model)
#     BestModel = SVMModels.best_estimator


# step 4: Evaluate the best model on test set
T0 = time.time()
y_pred = SVMModels.predict(x_test)
print("The testing time for random split classification is %0.3f sec." % (round(time.time() - T0, 2)))
Accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy = {}".format(Accuracy))
print(metrics.classification_report(y_test,
                                    y_pred, labels=[1, 0],
                                    target_names=['Malware', 'Goodware']))
Report = "Test Set Accuracy = " + str(Accuracy) + "\n" + metrics.classification_report(y_test,
                                                                                       y_pred,
                                                                                       labels=[1, -1],
                                                                                       target_names=['Malware',
                                                                                                     'Goodware'])

forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)

forest.fit(x_, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
# print("Feature ranking:")

# for f in range(x_.shape[1]):
#     print("%d. feature %s , %d (%f)" % (f + 1, features[f],indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_.shape[1]), indices)
plt.xlim([-1, x_.shape[1]])
# plt.show()


portions=[0.05, 0.1]
for portion in portions:
    sfm = SelectFromModel(forest ,prefit=True,  threshold=importances[indices[int(portion*len(indices))]])#.transform(x_)
    # sfm = SelectFromModel(forest ,prefit=True,  threshold=0.0001428)#.transform(x_)
    # sfm = SelectPercentile(forest, percentile=10)
    # sfm.fit(x_, y)

    # print(importances>importances[int(0.1*len(indices))])
    # features_map = {x: 0 for x in range(1, 9)}
    features_map = {x: 0 for x in FEATURES_SET}
    Header = ['importance', 'feature', 'drebin_category']
    print("Writing data ...")
    results=[]
    with open('/home/maryam/Code/data/'+str(portion)+'/results_.csv', 'w',newline='') as outfile1:
        wr = csv.writer(outfile1, delimiter=',', quoting=csv.QUOTE_NONE)
        wr.writerow([h for h in Header])

        sfmsupport=sfm.get_support("prefit=True")
        for f in range(len(sfmsupport)):
            # print("  %d: (%.10f) (%s)" % (f + 1,importances[sfmsupport[f]], features[sfmsupport[f]]))
            if features[f] != "":
                set = features[sfmsupport[f]].split("::")[0]
                features_map[set] += 1
                result=[importances[sfmsupport[f]], features[sfmsupport[f]], set]
                results.append(result)
                wr.writerow(result)
    outfile1.close()
    np.save('/home/maryam/Code/data/'+str(portion)+'/Drebin_important_features.npy',results)
    print(features_map)
    X_important_train = sfm.transform(x_)
    print(X_important_train.shape)


    x_train, x_test, y_train, y_test = train_test_split(X_important_train, y, test_size=TestSize,
                                                                                random_state=random.randint(0, 100))

    scipy.sparse.save_npz('/short/yq94/feature_selection/data/drebin/'+str(portion)+'/X_important.npz', X_important_train)
    np.save('/home/maryam/Code/data/'+str(portion)+'/y.npy', y)

    scipy.sparse.save_npz('/short/yq94/feature_selection/data/drebin/'+str(portion)+'/x_train.npz', x_train)
    scipy.sparse.save_npz('/short/yq94/feature_selection/data/drebin/'+str(portion)+'/x_test.npz', x_test)
    np.save('/short/yq94/feature_selection/data/drebin/'+str(portion)+'/y_train.npy', y_train)
    np.save('/short/yq94/feature_selection/data/drebin/'+str(portion)+'/y_test.npy', y_test)
# x_train, x_test, y_train, y_test = train_test_split(X_important_train, y, test_size=TestSize,
#                                                                             random_state=random.randint(0, 100))
    print("Perform Classification with SVM Model")
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    T0 = time.time()
    # if not Model:
    Clf = GridSearchCV(LinearSVC(), Parameters, cv=5, scoring='f1', n_jobs=-1)
    SVMModels = Clf.fit(x_train, y_train)
    print(
        "Processing time to train and find best model with GridSearchCV is %s sec." % (round(time.time() - T0, 2)))
    BestModel = SVMModels.best_estimator_
    print("Best Model Selected : {}".format(BestModel))
    print("The training time for random split classification is %s sec." % (round(time.time() - T0, 2)))
    print("Enter a filename to save the model:")
    # filename = raw_input()
    filename = "SVM"
    # dump(Clf, filename + ".pkl")
    # step 4: Evaluate the best model on test set
    T0 = time.time()
    y_pred = SVMModels.predict(x_test)
    print("The testing time for random split classification is %0.3f sec." % (round(time.time() - T0, 2)))
    Accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = {}".format(Accuracy))
    print(metrics.classification_report(y_test,
                                        y_pred, labels=[1, 0],
                                        target_names=['Malware', 'Goodware']))
    Report = "Test Set Accuracy = " + str(Accuracy) + "\n" + metrics.classification_report(y_test,
                                                                                           y_pred,
                                                                                           labels=[1, -1],
                                                                                           target_names=['Malware',
                                                                                                         'Goodware'])



# plt.figure(figsize=(12, 8))
# plt.title("Score")
# plt.barh(indices, features_map, .2, label="", color='navy')
# plt.yticks(())
# plt.legend(loc='best')
# plt.subplots_adjust(left=.25)
# plt.subplots_adjust(top=.95)
# plt.subplots_adjust(bottom=.05)
#
# for i, c in zip(indices, clf_names):
#     plt.text(-.3, i, c)
#
# plt.show()
