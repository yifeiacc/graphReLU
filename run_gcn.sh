
echo "Cora"
echo "====="
python gcn.py --dataset=Cora  --random_splits=True 

echo "CiteSeer"
echo "====="
python gcn.py --dataset=CiteSeer  --random_splits=True 

echo "PubMed"
echo "====="
python gcn.py --dataset=PubMed  --random_splits=True 
