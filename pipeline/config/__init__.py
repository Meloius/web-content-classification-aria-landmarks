import pandas as pd, math

from sklearn import tree, svm, ensemble, multiclass, neighbors
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


seed = 42

def get_pipeline (classifier_name, ncol):
    pipeline = None
    params = {}
    if (classifier_name == 'svm'):
        classifier = svm.SVC(random_state=seed, probability=True)
        params = {
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            #'classifier__kernel': ['linear'],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__degree': [2, 3],
            #'classifier__coef0': [1, 10, 100],
            #'classifier__tol': [0.001, 0.1, 1],
            'classifier__class_weight': ['balanced']
        }
    elif (classifier_name == 'dt'):
        classifier = tree.DecisionTreeClassifier(random_state=seed)
        params = {
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [5, 7, 10, 50, None],
            'classifier__min_samples_split': [2, 3, 5, 7, 10],
            'classifier__class_weight': ['balanced']
        }
    elif (classifier_name == 'ovr_dt'):
        classifier = multiclass.OneVsRestClassifier(tree.DecisionTreeClassifier(random_state=seed))
        params = {
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [10, 50, 100, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': ['balanced']
        }
    elif (classifier_name == 'rf'):
        classifier = ensemble.RandomForestClassifier(random_state=seed)
        params = {
            'classifier__n_estimators': [5, 10, 20, 30],
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [5, 7, 10, 50, None],
            'classifier__min_samples_split': [2, 3, 5, 7, 10],
            'classifier__class_weight': ['balanced']
        }
    elif (classifier_name == 'ovr_rf'):
        classifier = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(random_state=seed))
        params = {
            'classifier__n_estimators': [5, 10, 100],
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [10, 50, 100, None],
            'classifier__min_samples_split': [2, 10, 100],
            'classifier__class_weight': ['balanced']
        }
    elif (classifier_name == 'knn'):
        classifier = neighbors.KNeighborsClassifier()
        params = {
            'classifier__n_neighbors': [5, 10, 20],
            'classifier__weights': ["uniform", "distance"],
            'classifier__leaf_size': [10, 30, 50],
            'classifier__p': [1, 2]
        }
    else:
        classifier = svm.LinearSVC(random_state=seed, fit_intercept=False)
        params = {
            #'classifier__penalty': ['l1', 'l2'],
            #'classifier__loss': ['hinge', 'squared_hinge'],
            'classifier__C': [0.1, 1, 5, 10],
            'classifier__class_weight': ['balanced'],
            'classifier__fit_intercept': [False],
            'classifier__dual': [False],
            'classifier__max_iter': [100000]
        }

    pipeline = Pipeline([
        ('variance', VarianceThreshold()),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('classifier', classifier),
    ])
    params.update({ 'selector__k': [ int(ncol*0.1), int(ncol*0.2), int(ncol*0.3), int(ncol*0.4), 'all' ] })
    folds = GroupShuffleSplit(3, random_state=42)
    return GridSearchCV(pipeline, params, cv=folds, error_score=0)


class EnhancingFeatures:

    def __init__ (self, columns_index, relative=True, classnames=True):
        self._vectorizer = CountVectorizer(binary=True)
        self._columns_index = columns_index
        self._new_features = None
        self._relative = relative
        self._classnames = classnames

    def fit (self, X, y=None):

        if self._classnames:
            label_index = self._columns_index.index('label')
            labels = [ str(label).lower().replace('_', ' ') if str(label) != 'nan' else '' for label in X[:, label_index] ]
            self._vectorizer.fit(labels)

        return self

    def transform (self, X, y=None):
        dataset = pd.DataFrame(X, columns=self._columns_index)

        if self._relative:
            count_features = [ attr for attr in self._columns_index if attr.endswith('_count') and attr != 'window_elements_count' ]
            childs_count_normalized = dataset.loc[:, 'childs_count'].replace(to_replace=0, value=1)

            df_counters_normalized = dataset.loc[:, count_features].divide(childs_count_normalized, axis=0)
            df_counters_normalized.columns = [ ('%s_reg' % (feature)) for feature in count_features ]
            dataset = pd.concat([dataset, df_counters_normalized], axis=1)

            dataset.loc[:, 'top_reg'] = dataset.loc[:, 'top'] / dataset.loc[:, 'window_height']
            dataset.loc[:, 'avg_top_reg'] = dataset.loc[:, 'avg_top'] / dataset.loc[:, 'window_height']
            dataset.loc[:, 'sd_top_reg'] = dataset.loc[:, 'sd_top'] / dataset.loc[:, 'window_height']
            dataset.loc[:, 'bottom_reg'] = dataset.loc[:, 'window_height'] - (dataset.loc[:, 'top'] + dataset.loc[:, 'height'])
            dataset.loc[:, 'right_reg'] = 1080 - (dataset.loc[:, 'left'] + dataset.loc[:, 'width'])

        if self._classnames:
            labels = [ str(label).lower() if str(label) != 'nan' else '' for label in dataset.loc[:, 'label'] ]
            result = self._vectorizer.transform(labels)
            vectors_df = pd.DataFrame(result.toarray().tolist(), columns=self._vectorizer.get_feature_names())
            dataset = pd.concat([dataset, vectors_df], axis=1)

        features = [ column for column in dataset.columns if column not in ['className', 'label'] ]
        if self._new_features is None:
            self._new_features = features

        return dataset.loc[:, features].to_numpy()

    def fit_transform (self, X, y=None):
        return self.fit(X, y).transform(X)

