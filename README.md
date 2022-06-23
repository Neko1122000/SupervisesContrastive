python main_supcon.py --batch_size 32 --model resnet34 --learning_rate 0.5 --temp 0.1 --num_worker 2 --epochs 10 --size 64 --dataset path

python validation_by_linear_model.py --batch_size 32 --model resnet34 --learning_rate 0.5 --num_worker 2 --epochs 10 --size 128 --temp 0.1

nohup <anaconda python path>/bin/python file.py --argument > log_file_path.txt &
nohup /home/trongld/miniconda3/envs/sup_con/bin/python main_supcon.py --batch_size 8 --model resnet34 --learning_rate 0.05 --num_worker 2 --epochs 15 --size 200 --dataset path > sup_con_out.log &