python main_supcon.py --batch_size 32 --model resnet34 --learning_rate 0.5 --temp 0.1 --num_worker 2 --epochs 10 --size 64 --dataset path

python validation_by_linear_model.py --batch_size 32 --model resnet34 --learning_rate 0.5 --num_worker 2 --epochs 10 --size 128 --temp 0.1

nohub <anaconda python path>/bin/python file.py --argument > log_file_path.txt &