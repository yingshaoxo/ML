# Can we play paddlepaddle for fun?

## install pandle
https://github.com/yingshaoxo/TechNotes/blob/main/2022/3/9/conda%20install%20paddlepaddle%20at%20m1%20macOS.md

## get latent vector
```bash
cd PaddleGAN/applications/
python -u tools/pixel2style2pixel.py \
       --input_image '/Users/yingshaoxo/CS/ML/15.PaddleGAN/PaddleGAN/theMe.jpeg' \
       --output_path '/Users/yingshaoxo/CS/ML/15.PaddleGAN/PaddleGAN/the_output' \
       --model_type ffhq-inversion \
       --seed 233 \
       --size 1024 \
       --style_dim 512 \
       --n_mlp 8 \
       --channel_multiplier 2 
```