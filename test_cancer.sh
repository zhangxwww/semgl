device=0

for i in 3 5 8 42 4968
do 
rm -rf ../out/breast_cancer/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/breast_cancer/idgl.yml --multi_run --method graph_update --update_lambda1 5e-5 --update_lambda2 5e-4 --update_lambda3 1e-1 --update_graph_alpha 5e-2 --update_steps 50 --beta 0 --lr 1e-2 --ce_term --eps_adj 1e-7 --random_seed "$i"
done

