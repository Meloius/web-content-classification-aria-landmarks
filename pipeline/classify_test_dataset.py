import pandas as pd, pickle, os

model = 'rf-True-False.sav'
THRESHOLD = 0.5

def classify_test ():

    partial_files = [ path for path in os.listdir('./data/test/') if path.endswith('-test.csv') ]

    extractor = pickle.load(open('./results/classifier/extractor-%s' % (model), 'rb'))
    pipeline = pickle.load(open('./results/classifier/pipeline-%s' % (model), 'rb'))

    landmarks = ['banner', 'main', 'contentinfo', 'form', 'navigation', 'search', 'region', 'complementary']

    for path in partial_files:
        classified = {}
        dataset = pd.read_csv('./data/test/%s' % (path))
        features = [ column for column in dataset.columns.tolist() if column not in ['Unnamed: 0', 'url', 'tagName', 'role', 'class', 'parent_landmark', 'screenshot', 'xpath'] ]

        X_test = dataset.loc[:, features].to_numpy()
        X_test = extractor.transform(X_test)
        y_pred = pipeline.predict_proba(X_test)
        y_classes = pipeline.predict(X_test)
        classes = pipeline.best_estimator_.named_steps['classifier'].classes_

        dataset.loc[:, 'class'] = y_classes
        for i, cl in enumerate(classes):
            dataset.loc[:, cl] = y_pred[:, i]

        for landmark in landmarks:
            if landmark not in classified:
                classified[landmark] = dataset.loc[dataset['class'] == landmark, :]
            else:
                classified[landmark] = classified[landmark].append(dataset.loc[dataset['class'] == landmark, :], ignore_index=True)

        dataset.to_csv('./results/test/%s' % (path))

        del dataset

        for landmark in landmarks:
            filename = './results/test/classified-%s.csv' % (landmark)
            df = classified[landmark]

            df = df.loc[df[landmark] >= THRESHOLD]
            #if landmark not in ['banner', 'contentinfo', 'main']:
            #    df = df.loc[df[landmark] >= THRESHOLD]
            #else:
            #    df = df.loc[df[landmark] == df[landmark].max()]

            if not os.path.exists(filename):
                df.to_csv(filename)
            else:
                df.to_csv(filename, mode='a', header=False)

        del classified
