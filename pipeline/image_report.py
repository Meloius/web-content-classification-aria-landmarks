import os, pandas as pd

from PIL import Image, ImageDraw
from urllib.parse import urlparse

error_webpages = ['114-437', '156-142', '176-120', '211-44', '389-317', '4-95', '408-178', '415-182', '67-169', '73-229', '81-273']

def generate_reports ():
    reports = [ report for report in os.listdir('./results/clusters/') if report.endswith('.csv') ]

    for report in reports:
        landmark, url_index, pag, _ = (report[0:-4]).split('-')

        xpath = report.endswith('.xpath.csv')

        df = pd.read_csv('./results/clusters/%s' % (report))
        nrows, _ = df.shape
        domain = df['url'].tolist()[0]
        target_screenshot = df['screenshot'].tolist()[0]
        url_hash = target_screenshot.split('/').pop()[:-4]
        target_screenshot = './data/screenshots/%s.png' % (url_hash)

        if '.json_.png' in target_screenshot or '.png_' in target_screenshot or url_hash in error_webpages:
            print('Webpage with error: 404, forbidden, impeditive modal, captcha, no CSS website')
            pass
        else:
            print('reporting %s -> %s -> %s' % (report, domain, target_screenshot))
            drawed_images = False
            try:
                with Image.open(target_screenshot) as img:
                    (width, height) = img.size
                    baseline = img.copy()

                    for i in range(nrows):
                        landmark_row = df.iloc[i, :]
                        top = landmark_row['top']
                        left = landmark_row['left']
                        height = landmark_row['height']
                        width = landmark_row['width']

                        if top >= 0 and left >= 0 and height >= 0 and width >= 0:
                            drawed_images = True
                            draw = ImageDraw.Draw(img)
                            draw.rectangle([(left, top), (left + width, top + height)], outline=(255, 0, 0, 255), width=3, fill=(0, 0, 255, 100))
                            draw.text((left + 10, top + 10), str(i), fill=(255, 0, 0))

                    url_hash = target_screenshot.split('/').pop()[:-4]
                    if not os.path.isdir('./results/image-reports/%s' % (url_hash)):
                        os.mkdir('./results/image-reports/%s' % (url_hash))
                        baseline.save('./results/image-reports/%s/baseline.png' % (url_hash))
                    if drawed_images:
                        xpath_str = 'xpath' if xpath else 'baseline'
                        img.save('./results/image-reports/%s/%s-%s-%s-%s-report.png' % (url_hash, landmark, url_index, pag, xpath_str))
                        df.loc[:, ['url', 'screenshot', 'xpath']].to_csv('./results/image-reports/%s/%s-%s-%s-%s.csv' % (url_hash, landmark, url_index, pag, xpath_str))
            except:
                print('no screenshot image -> means website with no CSS as well')
