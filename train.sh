echo "gcn"
echo "==================="

python gcn.py --dataset Cora --normalize_features true --runs 10 --random_splits true
python gcn.py --dataset PubMed --normalize_features true --runs 10  --random_splits true
python gcn.py --dataset CiteSeer --normalize_features true --runs 10  --random_splits true
python gcn.py --dataset flickr --normalize_features False --run 10 --random_splits true
python gcn.py --dataset BlogCatalog --normalize_features False --run 10 --random_splits true


echo "sage"
echo "==================="
python sage.py --dataset Cora --normalize_features true --runs 10 --random_splits true
python sage.py --dataset PubMed --normalize_features true --runs 10 --random_splits true
python sage.py --dataset CiteSeer --normalize_features true --runs 10 --random_splits true
python sage.py --dataset flickr --normalize_features False --random_splits true
python sage.py --dataset BlogCatalog --normalize_features False --run 10 --random_splits true

echo "Cheb"
echo "==================="
python cheb.py --dataset Cora --normalize_features true --runs 10 --random_splits true
python cheb.py --dataset PubMed --normalize_features true --runs 10 --random_splits true
python cheb.py --dataset CiteSeer --normalize_features true --runs 10 --random_splits true
python cheb.py --dataset flickr --normalize_features False --run 10 --random_splits true
python cheb.py --dataset BlogCatalog --normalize_features False --run 10 --random_splits true


echo "Arma"
echo "==================="
python arma.py --dataset Cora --normalize_features true --runs 10 --random_splits true
python arma.py --dataset PubMed --normalize_features true --runs 10 --random_splits true
python arma.py --dataset CiteSeer --normalize_features true --runs 10 --random_splits true
python arma.py --dataset flickr --normalize_features False --runs 10 --random_splits true
python arma.py --dataset BlogCatalog --normalize_features False --runs 10 --random_splits true


echo "Gat"
echo "==================="
python gat.py --dataset Cora --normalize_features true --runs 10 --random_splits true
python gat.py --dataset PubMed --normalize_features true --runs 10 --random_splits true
python gat.py --dataset CiteSeer --normalize_features true --runs 10 --random_splits true
python gat.py --dataset flickr --normalize_features False --runs 10 --random_splits true
python gat.py --dataset BlogCatalog --normalize_features False --runs 10 --random_splits true
