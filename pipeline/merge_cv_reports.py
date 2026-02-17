import pandas as pd, os, json

def merge_reports ():
    report_files = [ path for path in os.listdir('./results/classifier') if path.endswith('.json') ]
    reports = {}
    for filename in report_files:
        with open('./results/classifier/%s' % (filename)) as f:
            report = json.load(f)
            approach = filename[7:-5] # report-(%s).json
            reports[approach] = report

    landmarks = ['banner', 'complementary', 'contentinfo', 'form',
                 'main', 'navigation', 'search', 'region', 'macro avg', 'weighted avg']
    all_report = {}
    summary = {}
    for approach in reports:
        classifier = approach.split('-')[0]
        summary['%s-f1-score' % (classifier)] = []
        summary['%s-precision' % (classifier)] = []
        summary['%s-recall' % (classifier)] = []

    with pd.ExcelWriter('./results/cv/accuracy.xlsx') as writer:
        for landmark in landmarks:
            merged_report = { 'precision': {}, 'recall': {}, 'f1-score': {} }
            for approach in reports:
                report = reports[approach][landmark]
                merged_report['precision'][approach] = report['precision']
                merged_report['recall'][approach] = report['recall']
                merged_report['f1-score'][approach] = report['f1-score']

                all_report_key = 'precision-%s' % (approach)
                if all_report_key in all_report:
                    all_report['precision-%s' % (approach)].extend(report['precision'])
                    all_report['recall-%s' % (approach)].extend(report['recall'])
                    all_report['f1-score-%s' % (approach)].extend(report['f1-score'])
                else:
                    all_report['precision-%s' % (approach)] = report['precision']
                    all_report['recall-%s' % (approach)] = report['recall']
                    all_report['f1-score-%s' % (approach)] = report['f1-score']

                classifier = approach.split('-')[0]
                summary['%s-f1-score' % (classifier)].append(sum(report['f1-score']) / len(report['f1-score']))
                summary['%s-precision' % (classifier)].append(sum(report['precision']) / len(report['precision']))
                summary['%s-recall' % (classifier)].append(sum(report['recall']) / len(report['recall']))

            precision_df = pd.DataFrame(merged_report['precision'])
            precision_df.to_csv('./results/cv/%s-precision.csv' % (landmark))
            recall_df = pd.DataFrame(merged_report['recall'])
            recall_df.to_csv('./results/cv/%s-recall.csv' % (landmark))
            fscore_df = pd.DataFrame(merged_report['f1-score'])
            fscore_df.to_csv('./results/cv/%s-f1-score.csv' % (landmark))

            fscore_df.columns = [ column.split('-')[0].upper() for column in fscore_df.columns ]
            fscore_df.to_excel(writer, sheet_name='%s-fscore' % (landmark))
            precision_df.columns = [ column.split('-')[0].upper() for column in precision_df.columns ]
            precision_df.to_excel(writer, sheet_name='%s-precision' % (landmark))
            recall_df.columns = [ column.split('-')[0].upper() for column in recall_df.columns ]
            recall_df.to_excel(writer, sheet_name='%s-recall' % (landmark))

        summary_df = pd.DataFrame(summary, index=landmarks)
        summary_df.to_excel(writer, sheet_name='summary')

        features_files = [ filename for filename in os.listdir('./results/cv') if filename.startswith('features-') ]
        features_df = pd.DataFrame()
        for filename in features_files:
            df = pd.read_csv('./results/cv/%s' % (filename))
            features_df = pd.concat([features_df, df])

        features_df = features_df.reindex(features_df.mean().sort_values(ascending=False).index, axis=1)
        features_df.to_excel(writer, sheet_name='features')

    (pd.DataFrame(all_report)).to_csv('./results/cv/all.csv')
