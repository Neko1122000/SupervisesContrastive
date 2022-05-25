python main_supcon.py --batch_size 256 --model resnet34 --learning_rate 0.5 --temp 0.1 --num_worker 2 --epochs 10

python validation_by_linear_model.py --batch_size 256 --model resnet34 --learning_rate 0.5 --num_worker 2 --epochs 10