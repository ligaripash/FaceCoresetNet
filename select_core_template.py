#import net
import torch
import os
import torchvision
#from face_alignment import align
import numpy as np
#import PIL
#from PIL import ImageFilter
from PIL import Image
from utils import dotdict
import config
import train_val_template as train_val
# adaface_models = {\
#     'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt",
#     'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
#     #:"pretrained/adaface_ir18_vgg2.ckpt",
#     'ir_18':'experiments/run_ir18_ms1mv2_subset_04-22_5/epoch=24-step=45650.ckpt'}

# def load_pretrained_model(architecture='ir_50'):
#     # load model and pretrained statedict
#     assert architecture in adaface_models.keys()
#     model = net.build_model(architecture)
#     statedict = torch.load(adaface_models[architecture])['state_dict']
#     model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
#     model.load_state_dict(model_statedict)
#     model.eval()
#     return model



def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img_hwc = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    #rgb_img = ((np_img / 255.) - 0.5) / 0.5
    #rgb = > rbg
    tensor_chw = torch.tensor([bgr_img_hwc.transpose(2,0,1)]).float()
    #tensor = torch.tensor([rgb_img.transpose(1, 2, 0)]).float()
    return tensor_chw

if __name__ == '__main__':

    center_crop = torchvision.transforms.CenterCrop(112)
    resize = torchvision.transforms.Resize((112,112))
    args = config.get_args()
    hparams = dotdict(vars(args))

    model = train_val.FaceCoresetNet(**hparams)

    checkpoint = torch.load(args.resume_from_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.aggregate_model.gamma = torch.nn.Parameter(torch.tensor(0.0))
    model.eval()


    test_image_path = 'D:/exp/FaceCoresetNet/qualitative_exp/messi'
    template = []
    file_list = sorted(os.listdir(test_image_path))
    for fname in file_list:
        path = os.path.join(test_image_path, fname)
        img = Image.open(path).convert('RGB')
        #img = center_crop(img)
        #aligned_rgb_img = align.get_aligned_face(path)
        aligned_rgb_img = resize(img)
        input = to_input(aligned_rgb_img).unsqueeze(1)
        #img = Image.open(path).convert('RGB')
        template.append(input)

    template_tensor = torch.cat(template, dim=1)
    unnorm_embeddings, FPS_sample = model(templates=template_tensor, labels=None, embeddings=None, norms=None, compute_feature=True, only_FPS=True)
    for i in range(unnorm_embeddings.shape[1]):
        for j in range(FPS_sample.shape[1]):
            if (unnorm_embeddings[0,i,:] == FPS_sample[0,j,:]).all():
                print('template index:' + str(i) + ' match FPS index:' + str(j))
                print('file is ' + file_list[i])

    pass

    

