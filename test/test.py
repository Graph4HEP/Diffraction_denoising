import os,sys
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
sys.path.append('../utils')
sys.path.append('../')
import utils
from dataset.dataset_denoise import *
from model import *


output_dir = sys.argv[1]
data_dir   = sys.argv[2]
samples    = sys.argv[3]
factor     = sys.argv[4]

if(samples=='full'):
    pass
else:
    try:
        samples = int(samples)
    except:
        print('argv 3 should be an int number or string "full". It is the sample number to test the input data.')
        sys.exit()


try:
    factor = int(factor)
except:
    print('argv 4 should be a int number, please try again. It will time the pixel value to have a better view of the image, the suggestion value is 4. But other value maybe even better.')
    sys.exit()

model_name = f'{output_dir}/models/model_best.pth'
result_dir = f'{output_dir}/results/'
config_dir = f'{output_dir}/config.json'
with open(config_dir, 'r') as f:
    config = json.load(f)

model_restoration = Uformer(img_size=config['train_ps'],embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=config['dd_in'])

utils.load_checkpoint(model_restoration, model_name)
model_restoration.cuda()
model_restoration.eval()

test_path_hc = f'{data_dir}/HC/'
test_path_lc = f'{data_dir}/LC/'
test_dataset_hc = get_test_data(test_path_hc)
test_dataset_lc = get_test_data(test_path_lc)
Nsample = len(test_dataset_lc) if samples=='full' else samples
with torch.no_grad():
    for i in tqdm(range(Nsample)):
        img = test_dataset_lc[i][0].reshape(1,3,256,256).to('cuda')
        out = model_restoration(img)
        out = torch.clamp(out,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        out = out[:,:,0]*255*factor
        image_hc = test_dataset_hc[i][0][0]*255*factor
        image_lc = test_dataset_lc[i][0][0]*255*factor
        image_hc = Image.fromarray(image_hc.numpy())
        image_lc = Image.fromarray(image_lc.numpy())
        image_out= Image.fromarray(out.numpy())
        image = Image.new('L', (788,256))
        image.paste(image_lc, (0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(256, 0), (266, 256)], fill='white')
        image.paste(image_hc, (266,0))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(522, 0), (532, 256)], fill='white')
        image.paste(image_out, (532, 0))
        plt.imshow(np.squeeze(image))
        plt.title(f"sample {i}, left to right: lc, hc, model output, multiplier factor {factor}")
        plt.show()
        plt.savefig(f'{result_dir}/test_result_{i}.png')
