import numpy as np
import torch
from lungmask import utils
import SimpleITK as sitk
from .resunet import UNet
import warnings
import sys
from tqdm import tqdm
import skimage
import logging
from abc import ABCMeta, abstractmethod

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# stores urls and number of classes of the models

# I will apply Structural Design pattern (bridge pattern) into this project5.
# I believe bridge pattern could make mask.py file and project better
# Comparing to the original code, I add two abstract class "Models" & "Applying_Model"
# For the abstract class "Models", it means it will get different type of models, such as "R231, LTRCLobe, R231CovidWeb..."
# In the future, it might be have more other accident disease, then we could easily add one models and inherited Models class's method "get_model"
# For the abstract class "Applying_Models", it  will have one abstract method called "choose_apply_way"
# Then there are two different applying model's way class "Apply" & "Apply_Fused" inherited abstract class "Applying_Models"
# These classes will override abstract method "chhose_apply_way". 
# Because this is the bridge pattern, when we “_init_”(Constructor), abstract class "Models" associated to abstract class "Applying_Models"
# Using bridge pattern will make the code and structure more clear and have a realationship.
# We could easily add more different type of models or different type of Applying way in the future 
# For example, in the future, we could add a smoking model about lungmask and a new specific smoking applying way in the project.


class Models(metaclass=ABCMeta):
    def __init__(self, applying_model):
        self._applying_model = applying_model

    model_urls = {('unet', 'R231'): ('https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth', 3),
              ('unet', 'LTRCLobes'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth', 6),
              ('unet', 'R231CovidWeb'): (
                  'https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth', 3)}
    
    def get_model(modeltype, modelname):
        model_url, n_classes = Models.model_urls[(modeltype, modelname)]
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device('cpu'))
        if modeltype == 'unet':
            model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
        elif modeltype == 'resunet':
            model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=True)
        else:
            logging.exception(f"Model {modelname} not known")
        model.load_state_dict(state_dict)
        model.eval()
        return model

        
class R231Model(Models):
    def get_model(modeltype):
        super().get_model()

class LTRCLobesModel(Models):
    def get_model(modeltype):
        super().get_model()            

class R231CovidWebModel(Models):
    def get_model(modeltype):
        super().get_model()

class LTRCLobes_R231Model(models):
    def get_model(modeltype):
        super().get_model()


class Applying_Model(Models):
    @abstractmethod
    def choose_apply_way(self):
        pass

class Apply(Applying_Model):
    # overriding abstract method
    def choose_apply_way(image, model=None, force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
        if model is None:
            r231 = R231Model()
            model = r231.get_model('unet', 'R231')
        
        numpy_mode = isinstance(image, np.ndarray)
        if numpy_mode:
            inimg_raw = image.copy()
        else:
            inimg_raw = sitk.GetArrayFromImage(image)
            directions = np.asarray(image.GetDirection())
            if len(directions) == 9:
                inimg_raw = np.flip(inimg_raw, np.where(directions[[0,4,8]][::-1]<0)[0])
        del image

        if force_cpu:
            device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
                batch_size = 1
                device = torch.device('cpu')
        model.to(device)

        
        if not noHU:
            tvolslices, xnew_box = utils.preprocess(inimg_raw, resolution=[256, 256])
            tvolslices[tvolslices > 600] = 600
            tvolslices = np.divide((tvolslices + 1024), 1624)
        else:
            # support for non HU images. This is just a hack. The models were not trained with this in mind
            tvolslices = skimage.color.rgb2gray(inimg_raw)
            tvolslices = skimage.transform.resize(tvolslices, [256, 256])
            tvolslices = np.asarray([tvolslices*x for x in np.linspace(0.3,2,20)])
            tvolslices[tvolslices>1] = 1
            sanity = [(tvolslices[x]>0.6).sum()>25000 for x in range(len(tvolslices))]
            tvolslices = tvolslices[sanity]
        torch_ds_val = utils.LungLabelsDS_inf(tvolslices)
        dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1,
                                                    pin_memory=False)

        timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

        with torch.no_grad():
            for X in tqdm(dataloader_val):
                X = X.float().to(device)
                prediction = model(X)
                pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
                timage_res = np.vstack((timage_res, pls))

        # postprocessing includes removal of small connected components, hole filling and mapping of small components to
        # neighbors
        if volume_postprocessing:
            outmask = utils.postrocessing(timage_res)
        else:
            outmask = timage_res

        if noHU:
            outmask = skimage.transform.resize(outmask[np.argmax((outmask==1).sum(axis=(1,2)))], inimg_raw.shape[:2], order=0, anti_aliasing=False, preserve_range=True)[None,:,:]
        else:
            outmask = np.asarray(
                [utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
                dtype=np.uint8)
        
        if not numpy_mode:
            if len(directions) == 9:
                outmask = np.flip(outmask, np.where(directions[[0,4,8]][::-1]<0)[0])    
        
        return outmask.astype(np.uint8)


class Apply_Fused(Applying_Model):
    # overriding abstract method
    def choose_apply_way(image, basemodel = 'LTRCLobes', fillmodel = 'R231', force_cpu=False, batch_size=20, volume_postprocessing=True, noHU=False):
        '''Will apply basemodel and use fillmodel to mitiage false negatives'''
        ltrclobes = LTRCLobesModel()
        mdl_r = ltrclobes.get_model('unet',fillmodel)
        mdl_l = ltrclobes.get_model('unet',basemodel)
        logging.info("Apply: %s" % basemodel)
        res_l = apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
        logging.info("Apply: %s" % fillmodel)
        res_r = apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
        spare_value = res_l.max()+1
        res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
        res_l[res_r==0] = 0
        logging.info("Fusing results... this may take up to several minutes!")
        return utils.postrocessing(res_l, spare=[spare_value])