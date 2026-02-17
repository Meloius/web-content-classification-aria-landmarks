import pandas as pd

CHUNKSIZE = 50000

landmarks = ['banner', 'complementary', 'contentinfo', 'form',
             'navigation', 'search', 'region', 'main']
def cluster_rows ():
    banner = None
    contentinfo = None
    complementary = None

    for landmark in landmarks:
        nrows = CHUNKSIZE
        with pd.read_csv('./results/test/classified-%s.csv' % (landmark), iterator=True) as parser:
            chunk_count = 0
            while nrows == CHUNKSIZE:
                chunk_count = chunk_count + 1
                df = parser.get_chunk(CHUNKSIZE)
                nrows, _ = df.shape
                urls = pd.unique(df['url']).tolist()

                for url_ind, url in enumerate(urls):
                    clusters = []
                    df_url = df.loc[df['url'] == url, :]
                    nrows_url, _ = df_url.shape

                    for i in range(nrows_url-1, -1, -1):
                        target = df_url.iloc[i, :]
                        left, right, top, bottom = target['left'], target['left'] + target['width'], target['top'], target['top'] + target['height']
                        count = 0

                        includes_banner = True
                        includes_contentinfo = True

                        if banner is not None and (target['top'] >= (banner['top'] + banner['height'])):
                            includes_banner = False
                        if contentinfo is not None and (target['top'] + target['height']) <= contentinfo['top']:
                            includes_contentinfo = False

                        for cluster in clusters:
                            if (top >= cluster['bottom'] or bottom <= cluster['top'] or
                                left >= cluster['right'] or right <= cluster['left']): # no insersection
                                count = count + 1
                            else:
                                cluster['left'] = min(left, cluster['left'])
                                cluster['right'] = max(right, cluster['right'])
                                cluster['top'] = min(top, cluster['top'])
                                cluster['bottom'] = max(bottom, cluster['bottom'])

                                if landmark != 'main':
                                    if cluster['higher_probability'][landmark] < target[landmark]:
                                        cluster['higher_probability'] = target
                                        cluster['includes_banner'] = includes_banner
                                        cluster['includes_contentinfo'] = includes_contentinfo
                                else:
                                    if cluster['higher_probability'][landmark] < target[landmark] and not includes_banner and not includes_contentinfo:
                                        cluster['higher_probability'] = target
                                        cluster['includes_banner'] = includes_banner
                                        cluster['includes_contentinfo'] = includes_contentinfo
                                break

                        if count == len(clusters):
                            if landmark != 'main' or (not includes_banner and not includes_contentinfo):
                                clusters.append({
                                    'higher_probability': target,
                                    'left': left,
                                    'right': right,
                                    'top': top,
                                    'bottom': bottom,
                                    'includes_banner': includes_banner,
                                    'includes_contentinfo': includes_contentinfo
                                })

                        for i in range(len(clusters)): # merge clusters
                           cluster_1 = clusters[i]
                           to_remove = []
                           for j in range(i+1, len(clusters)):
                               cluster_2 = clusters[j]
                               if (cluster_1['top'] >= cluster_2['bottom'] or
                                   cluster_1['bottom'] <= cluster_2['top'] or
                                   cluster_1['left'] >= cluster_2['right'] or
                                   cluster_1['right'] <= cluster_2['left']): # no insersection
                                   pass
                               else:
                                   clusters[i]['left'] = min(clusters[i]['left'], clusters[j]['left'])
                                   clusters[i]['right'] = max(clusters[i]['right'], clusters[j]['right'])
                                   clusters[i]['top'] = min(clusters[i]['top'], clusters[j]['top'])
                                   clusters[i]['bottom'] = max(clusters[i]['bottom'], clusters[j]['bottom'])

                                   if (cluster_1['higher_probability'][landmark] < cluster_2['higher_probability'][landmark]):
                                       clusters[i]['higher_probability'] = cluster_2['higher_probability']

                                   to_remove.append(j)

                           to_remove.reverse()
                           for remove_index in to_remove:
                                clusters.pop(remove_index)

                           if i + 1 >= len(clusters): break


                    result = pd.DataFrame([])
                    for cluster in clusters:
                        result = pd.concat([result, pd.DataFrame([cluster['higher_probability']])])
                        if landmark == 'banner':
                            banner = cluster['higher_probability']
                        if landmark == 'contentinfo':
                            contentinfo = cluster['higher_probability']
                    if (result.size > 0):
                        result.to_csv('./results/clusters/%s-%d-%d-results.csv' % (landmark, url_ind, chunk_count))


