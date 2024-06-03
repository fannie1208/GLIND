python main.py --dataset cifar10 --weight_ 0.0005 --context graph --store
python main.py --dataset cifar10 --back gat --weight_ 0.0005 --context graph --store
python main.py --dataset cifar10 --back difformer --weight_ 0.0005 --context graph --use_residual --use_weight --store
python main.py --dataset stl --weight_decay 0.001 --context graph --store
python main.py --dataset stl --back gat --weight_decay 0.001 --context graph --store
python main.py --dataset stl --back difformer --K 4 --weight_decay 0.001 --context graph --use_residual --store