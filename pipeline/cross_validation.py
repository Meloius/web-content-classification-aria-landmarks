import logging, sys, numpy as np, pandas as pd, json, pickle

from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit

from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss
from imblearn.over_sampling import SMOTE

from pipeline.config import get_pipeline, EnhancingFeatures

def fit_classifier (classifier):
    sampler = 'None'
    relative_arg = 'True'
    classname_arg = 'False'

    relative_arg = True if relative_arg == 'True' else False
    classname_arg = True if classname_arg == 'True' else False

    dataset = pd.read_csv('./data/training.classified.csv')
    features = [ column for column in dataset.columns.tolist() if column not in ['Unnamed: 0', 'url', 'tagName', 'role', 'class', 'parent_landmark', 'screenshot', 'xpath'] ]
    new_features = None

    run_cv = True
    extractor = EnhancingFeatures(features, relative=relative_arg, classnames=classname_arg)

    X = dataset.loc[:, features].to_numpy()
    y = dataset.loc[:, 'class'].to_numpy()
    groups = dataset.loc[:, 'url'].to_numpy()

    folds = GroupShuffleSplit(n_splits=10, random_state=42)
    y_pred = []
    y_true = []
    reports = []
    selected_features = []

    class NoneSampler:
        def fit_resample(self, X, y):
            nrows, _ = X.shape
            self.sample_indices_ = list(range(nrows))
            return X, y

    if sampler == 'SMOTE':
        sampler = SMOTE()
    elif sampler == 'tomek':
        sampler = TomekLinks()
    elif sampler == 'nearmiss1':
        sampler = NearMiss(sampling_strategy='majority', version=1)
    elif sampler == 'nearmiss2':
        sampler = NearMiss(sampling_strategy='majority', version=2)
    elif sampler == 'nearmiss3':
        sampler = NearMiss(sampling_strategy='majority', version=3)
    else:
        sampler = NoneSampler()

    if run_cv:
        for train, test in folds.split(X, y, groups):
            extractor = EnhancingFeatures(features, relative=relative_arg, classnames=classname_arg)

            X_train, y_train = X[train, :], y[train]
            X_test, y_test = X[test, :], y[test]
            groups_train = groups[train]

            X_train = extractor.fit_transform(X_train)
            X_test = extractor.transform(X_test)

            X_sample, y_sample = sampler.fit_resample(X_train, y_train)
            groups_sample = groups[sampler.sample_indices_]

            pipeline = get_pipeline(classifier, X_sample.shape[1])
            pipeline.fit(X_sample, y_sample, groups=groups_sample)

            result = pipeline.predict(X_test)
            y_pred.extend(result)
            y_true.extend(y_test)
            #print(pipeline.best_params_)
            variance_threashold, _, selector, _ = pipeline.best_estimator_.steps

            variance = variance_threashold[1].get_support()
            new_features = [ feature for i, feature in enumerate(extractor._new_features) if variance[i] ]

            kbest = selector[1].get_support()
            selected = [ feature for i, feature in enumerate(new_features) if kbest[i] ]
            #print(selected)

            selected_features.append([ True if i in selected else False for i in extractor._new_features ])

            reports.append(metrics.classification_report(result, y_test, output_dict=True))


    print('  ===== Training =====')
    pipeline = get_pipeline(classifier, X.shape[1])
    X = extractor.fit_transform(X)
    X, y = sampler.fit_resample(X, y)
    groups = groups[sampler.sample_indices_]
    pipeline.fit(X, y, groups=groups)
    pickle.dump(extractor, open('./results/classifier/extractor-%s-%s-%s.sav' % (classifier, relative_arg, classname_arg), 'wb'))
    pickle.dump(pipeline, open('./results/classifier/pipeline-%s-%s-%s.sav' % (classifier, relative_arg, classname_arg), 'wb'))
    print(metrics.classification_report(y, pipeline.predict(X)))

    if run_cv:
        print('\n\n  ===== Cross-Validation =====')
        print(metrics.classification_report(y_true, y_pred))

        print(pipeline.best_params_)
        variance_threashold, _, selector, _ = pipeline.best_estimator_.steps

        variance = variance_threashold[1].get_support()
        new_features = [ feature for i, feature in enumerate(extractor._new_features) if variance[i] ]

        kbest = selector[1].get_support()
        #print([ feature for i, feature in enumerate(new_features) if kbest[i] ])

        csv_reports = {}
        for report in reports:
            for cl in report:
                if cl != 'accuracy':
                    if (cl not in csv_reports):
                        csv_reports[cl] = {}
                    cl_report = report[cl]
                    for metric in cl_report:
                        if (metric not in csv_reports[cl]):
                            csv_reports[cl][metric] = []
                        csv_reports[cl][metric].append(cl_report[metric])

        (pd.DataFrame(data=selected_features, columns=extractor._new_features)).to_csv('./results/classifier/features-%s-%s-%s.csv' % (classifier, relative_arg, classname_arg))
        with open('./results/classifier/report-%s-%s-%s.json' % (classifier, relative_arg, classname_arg), 'w') as f:
            f.write(json.dumps(csv_reports))
            f.close()
