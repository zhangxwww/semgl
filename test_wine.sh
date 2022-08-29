device=3

for i in 6600 6666 4968 2842 4800
do 
rm -rf ../out/wine/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/wine/idgl.yml --multi_run --method graph_update --update_lambda1 5.5e-3 --update_lambda2 5e-2 --update_lambda3 5e0 --update_graph_alpha 5e-2 --update_steps 20 --beta 0 --lr 1e-2 --ce_term --random_seed $i
done
