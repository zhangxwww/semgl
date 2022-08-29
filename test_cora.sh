device=2

for i in 42 448 854 29493 88867
do 
rm -rf ../out/cora/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/cora/idgl.yml --multi_run --method graph_update --update_lambda1 1e-6 --update_lambda2 1e-4 --update_graph_alpha 5e-2 --update_steps 1 --beta 0 --lr 1e-2 --random_seed $i
done
