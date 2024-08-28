python3 ./train/train_denoise.py --arch Uformer_B --batch_size 64 --gpu '0' \
    --train_ps 128 --train_dir /root/autodl-tmp/training --env _first_train \
    --val_dir /root/autodl-tmp/validation --save_dir ./logs/ \
    --dataset diffraction --warmup 
