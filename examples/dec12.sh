for dataset in rel-stackex
do
for task in rel-stackex-engage rel-stackex-votes
do

echo python xgboost_baseline.py --dataset $dataset --task $task
python xgboost_baseline.py --dataset $dataset --task $task

echo python baseline.py --dataset $dataset --task $task
python baseline.py --dataset $dataset --task $task

echo python gnn.py --dataset $dataset --task $task
python gnn.py --dataset $dataset --task $task

done
done
