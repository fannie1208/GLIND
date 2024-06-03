python main.py --dataset arxiv --r 0.01 --weight_decay 0.0001 --lamda 0.9 --dropout 0.2 --context node --variant --store
python main.py --dataset arxiv --back gat --r 0.01 --weight_decay 5e-5 --tau 2 --K 4 --lamda 1.1 --dropout 0.2 --context graph --store
python main.py --dataset twitch --r 0.01 --weight_decay 0.0005 --lamda 1.5 --dropout 0.1 --context graph --store
python main.py --dataset twitch --back gat --r 0.01 --weight_decay 5e-5 --tau 2 --lamda 1 --dropout 0 --context graph --store