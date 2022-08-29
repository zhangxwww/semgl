device=0

for i in 1234 6666 4968 2842 4800
do 
rm -rf ../out/20news/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/20news/idgl.yml --multi_run --method graph_update --update_lambda1 1e-3 --update_lambda2 1e-2 --update_graph_alpha 5e-2 --update_steps 1 --beta 0 --lr 1e-3 --random_seed "$i"
done

