python3 ./train/train_denoise.py --arch Uformer_B --batch_size 64 --gpu '0' \
    --train_ps 128 --train_dir ../example_data/training --env _test \
    --val_dir ../example_data/validation --save_dir ./logs/ \
    --dataset diffraction --warmup 
