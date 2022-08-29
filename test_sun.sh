device=4

for i in 42 448 854 29493 88867
do 
rm -rf ../out/sun/idgl
CUDA_VISIBLE_DEVICES=$device python3 -u -W ignore main.py -config config/sun/idgl.yml --multi_run --method graph_update --update_lambda1 5e-6 --update_lambda2 1e-4 --update_graph_alpha 5e-2 --update_steps 1 --beta 0 --lr 1e-3 --random_seed "$i"
done

