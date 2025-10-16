from bs4 import BeautifulSoup
import os

html_files = [f for f in os.listdir('orig') if f.endswith('html')]

for f in html_files:
    with open('orig/' + f, encoding='latin1') as fi:
        soup = BeautifulSoup(fi, 'html.parser')
        pre1 = soup.pre
        if pre1:
            with open(f[:-5] + '.anb', 'w', encoding='utf8') as fo:
                fo.write(pre1.get_text())
