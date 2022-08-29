device=1

for i in 41 78 57 70 10
do 
rm -rf ../out/citeseer/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/citeseer/idgl.yml --multi_run --method graph_update --update_lambda1 5e-9 --update_lambda2 1e-2 --update_graph_alpha 5e-2 --update_steps 1 --beta 0 --lr 5e-4 --random_seed "$i"
done
