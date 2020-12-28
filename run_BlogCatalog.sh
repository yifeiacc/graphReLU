echo "BlogCatalog"
echo "======"

echo "GCN"
# python gcn.py --dataset=flickr
python gcn.py --dataset=BlogCatalog --random_splits=True --normalize_features=False

echo "GAT"
# python gat.py --dataset=flickr --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --dataset=BlogCatalog --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits=True --normalize_features=False

echo "Cheby"
# python cheb.py --dataset=flickr --num_hops=2
python cheb.py --dataset=BlogCatalog --num_hops=2 --random_splits=True --normalize_features=False

echo "SGC"
# python sgc.py --dataset=flickr --K=2 --weight_decay=0.0005
python sgc.py --dataset=BlogCatalog --K=2 --weight_decay=0.0005 --random_splits=True --normalize_features=False

echo "ARMA"
# python arma.py --dataset=flickr --num_stacks=2 --num_layers=1 --skip_dropout=0
python arma.py --dataset=BlogCatalog --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits=True --normalize_features=False

echo "APPNP"
# python appnp.py --dataset=flickr --alpha=0.1
python appnp.py --dataset=BlogCatalog --alpha=0.1 --random_splits=True --normalize_features=False
