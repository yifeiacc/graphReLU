echo "gcn"
echo "==================="
# & D:/anaconda3/python.exe d:/workspace/graphRelu/gcn.py --dataset Cora --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/gcn.py --dataset PubMed --normalize_features true  --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/gcn.py --dataset CiteSeer --normalize_features true  --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/gcn.py --dataset flickr --normalize_features False --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/gcn.py --dataset BlogCatalog --normalize_features False --random_splits true


echo "Gat"
echo "==================="
& D:/anaconda3/python.exe d:/workspace/graphRelu/gat.py --dataset Cora --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/gat.py --dataset PubMed --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/gat.py --dataset CiteSeer --normalize_features true --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/gat.py --dataset flickr --normalize_features False --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/gat.py --dataset BlogCatalog --normalize_features False --random_splits true


echo "Cheb"
echo "==================="
& D:/anaconda3/python.exe d:/workspace/graphRelu/cheb.py --dataset Cora --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/cheb.py --dataset PubMed --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/cheb.py --dataset CiteSeer --normalize_features true --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/cheb.py --dataset flickr --normalize_features False --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/cheb.py --dataset BlogCatalog --normalize_features False --random_splits true


echo "Arma"
echo "==================="
& D:/anaconda3/python.exe d:/workspace/graphRelu/arma.py --dataset Cora --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/arma.py --dataset PubMed --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/arma.py --dataset CiteSeer --normalize_features true --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/arma.py --dataset flickr --normalize_features False --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/arma.py --dataset BlogCatalog --normalize_features False --random_splits true


echo "sage"
echo "==================="
& D:/anaconda3/python.exe d:/workspace/graphRelu/sage.py --dataset Cora --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/sage.py --dataset PubMed --normalize_features true --random_splits true
& D:/anaconda3/python.exe d:/workspace/graphRelu/sage.py --dataset CiteSeer --normalize_features true --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/sage.py --dataset flickr --normalize_features False --random_splits true
# & D:/anaconda3/python.exe d:/workspace/graphRelu/sage.py --dataset BlogCatalog --normalize_features False --random_splits true