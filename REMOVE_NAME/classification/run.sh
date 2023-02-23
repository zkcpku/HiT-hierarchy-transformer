# CUDA_VISIBLE_DEVICES=6 python run_HiT.py Transformer_Java_Train_HiT 2>&1| tee Transformer_Java_Train_HiT_abs.txt

# python run_HiT.py Transformer_c1400_Train_HiT

CUDA_VISIBLE_DEVICES=3 python run_HiT.py Transformer_python_Train_HiT 2>&1| tee Transformer_python_Train_HiT_abs.txt
