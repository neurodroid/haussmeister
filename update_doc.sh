#! /bin/sh

git checkout gh-pages
git merge master
cd doc
make html
cd ..
rsync -av ./doc/_build/html/* ./
git add .
git commit -m "Update documentation"     
git push origin gh-pages
git checkout master
