import os
import numpy as np
import torch
import torch.utils.data as data
import random
import natsort
import pdb
from common.utils_loader import rgb_read, resize_rgb_image, sn_image_from_npy, crop_image, thermal_read, depth_read, resize_sn_image

__all__ = ['M2P2']

# class M2P2(data.Dataset):

#     def __init__(self, cfg, mode="test"):
#         self.data_root      = cfg.data_root               # root of your dataset
#         self.raw_cam_img_size   = tuple(cfg.raw_cam_img_size)     # e.g. [480, 640]
#         self.ratio          = cfg.ratio                   # any down‑sampling factor
#         self.load_interval  = cfg.get("load_interval", 1)
#         self.mode           = mode                        # "train"/"valid"/"test"
#         self.num_samples    = getattr(cfg, f"num_{mode}_samples", -1)

#         # build a list of samples
#         img_dir = os.path.join(self.data_root, mode, "thermal")
#         sn_dir  = os.path.join(self.data_root, mode, "surface_normal")
#         fp_dir  = os.path.join(self.data_root, mode, "footprint")

#         files    = natsort.natsorted(os.listdir(img_dir))
#         samples  = []
#         for fn in files[::self.load_interval]:
#             base = os.path.splitext(fn)[0]
#             entry = {
#                 "therm":      os.path.join(img_dir, fn),
#                 "sn":         os.path.join(sn_dir,  base + ".npy"),
#                 "footprint":  os.path.join(fp_dir,  base + ".png"),
#                 "fname":      base
#             }
#             samples.append(entry)

#         if self.num_samples>0: samples = samples[:self.num_samples]
#         if mode=="train":  random.shuffle(samples)
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         S = self.samples[idx]

#         # 1) Loading thermal image
#         therm = rgb_read(S["therm"])[...,0]
#         therm = crop_image(therm, self.raw_cam_img_size)
#         therm = resize_rgb_image(therm, 
#                                 (self.raw_cam_img_size[0]//self.ratio,
#                                 self.raw_cam_img_size[1]//self.ratio))
        
#         therm = np.expand_dims(therm, 0) / 255.0

#         # 2) Loading surface normal 
#         sn_np = np.load(S["sn"])
#         sn_img = sn_image_from_npy(sn_np, self.raw_cam_img_size, px=3)
#         sn_img = resize_rgb_image(sn_img,
#                                   (self.raw_cam_img_size[0]//self.ratio,
#                                    self.raw_cam_img_size[1]//self.ratio))

#         sn_img = np.transpose(sn_img, (2,0,1)//127.0)

#         # 3) Stack onto a 4-channel tensor

#         inp = np.concatenate([therm, sn_img], 0)

#         fp = rgb_read(S["footprint"])[...,0]
#         fp = resize_rgb_image(fp,
#                             (self.raw_cam_img_size[0]//self.ratio,
#                              self.raw_cam_img_size[1]//self.ratio))
#         fp = (fp>0).astype(np.uint8)[None,...]

#         gts = {"sn": sn_img, "fp": fp}

#         return inp, gts, S["fname"]

class M2P2(data.Dataset):

    def __init__(self, cfg, mode="test"):
        self.cfg = cfg
        self.mode           = mode                        # "train"/"valid"/"test"
        self.data_root      = cfg.data_root               # root of your dataset
        self.raw_cam_img_size   = tuple(cfg.raw_cam_img_size)     # e.g. [480, 640]
        self.ratio          = cfg.ratio                   # any down‑sampling factor
        self.load_interval  = cfg.get("load_interval", 1)
        self.num_samples    = getattr(cfg, f"num_{mode}_samples", -1)

        # build a list of samples
        img_dir = os.path.join(self.data_root, mode, "thermal")
        depth_dir = os.path.join(self.data_root, mode, "depth")
        sn_dir  = os.path.join(self.data_root, mode, "surface_normal")
        fp_dir  = os.path.join(self.data_root, mode, "footprint")
        # pdb.set_trace() 
        files    = natsort.natsorted(os.listdir(img_dir))
        samples  = []
        for fn in files[::self.load_interval]:
            base = os.path.splitext(fn)[0]
            entry = {
                "therm":      os.path.join(img_dir, fn),
                "depth":      os.path.join(depth_dir, base + ".png"),
                "sn":         os.path.join(sn_dir,  base + ".npy"),
                "footprint":  os.path.join(fp_dir,  base + ".png"),
                "fname":      base
            }
            samples.append(entry)

        if self.num_samples>0: samples = samples[:self.num_samples]
        if mode=="train":  random.shuffle(samples)
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        S = self.samples[idx]

        ####### GPT
        # # 1) Loading thermal image
        # therm = rgb_read(S["therm"])[...,0]
        # therm = crop_image(therm, self.raw_cam_img_size)
        # therm = resize_rgb_image(therm, 
        #                         (self.raw_cam_img_size[0]//self.ratio,
        #                         self.raw_cam_img_size[1]//self.ratio))
        
        # therm = np.expand_dims(therm, 0) / 255.0

        # # 2) Loading surface normal 
        # sn_np = np.load(S["sn"])
        # sn_img = sn_image_from_npy(sn_np, self.raw_cam_img_size, px=3)
        # sn_img = resize_rgb_image(sn_img,
        #                           (self.raw_cam_img_size[0]//self.ratio,
        #                            self.raw_cam_img_size[1]//self.ratio))

        # sn_img = np.transpose(sn_img, (2,0,1)//127.0)

        # # 3) Stack onto a 4-channel tensor

        # inp = np.concatenate([therm, sn_img], 0)

        # fp = rgb_read(S["footprint"])[...,0]
        # fp = resize_rgb_image(fp,
        #                     (self.raw_cam_img_size[0]//self.ratio,
        #                      self.raw_cam_img_size[1]//self.ratio))
        # fp = (fp>0).astype(np.uint8)[None,...]

        # gts = {"sn": sn_img, "fp": fp}

        # return inp, gts, S["fname"]

        # GROKKK
        # Load thermal image (grayscale)
        # thermal = Image.open(sample["thermal"]).convert('L')  # Convert to grayscale
        # therm = thermal_read(S["therm"])
        therm = thermal_read(S["therm"])
        # pdb.set_trace()
        # print(therm.shape)
        therm_3ch = np.repeat(therm[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        print(therm_3ch.shape)
        therm_3ch = crop_image(therm_3ch, self.raw_cam_img_size)
        therm_3ch = resize_rgb_image(therm, 
                                (self.raw_cam_img_size[0]//self.ratio,
                                self.raw_cam_img_size[1]//self.ratio))

        # Repeat thermal channel to mimic RGB (3 channels)

        depth = depth_read(S["depth"])
        depth = crop_image(depth, self.raw_cam_img_size)
        depth = resize_rgb_image(depth, 
                                (self.raw_cam_img_size[0]//self.ratio,
                                self.raw_cam_img_size[1]//self.ratio))

        # Load depth (assuming NumPy array; adjust if different)
        # Ensure thermal and depth have matching dimensions
        # assert therm.shape == depth.shape, f"Size mismatch: thermal {therm.shape}, depth {depth.shape}"

        # Concatenate thermal (3 channels) and depth (1 channel) into rgbd
        # rgbd = np.concatenate([therm_3ch, depth[:, :, np.newaxis]], axis=2)  # Shape: (H, W, 4)
        therm_3ch = np.repeat(therm_3ch[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        print(f"Shape of therm_3ch: {therm_3ch.shape}")
        print(f"Shape of depth: {depth.shape}")
        rgbd = np.concatenate((therm_3ch, depth[:, :, np.newaxis]), axis=2)  # Shape: (H, W, 4)
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1)  # Shape: (4, H, W)

        # Load surface normals (assuming 3-channel NumPy array)
        sn = np.load(S["sn"])  # Shape: (H, W, 3) or (3, H, W)
        print(f"Shape of sn: {sn.shape}")
        sn_np = sn_image_from_npy(sn, self.raw_cam_img_size, px=3)
        sn_np = resize_sn_image(sn_np, (int(self.raw_cam_img_size[0]//self.ratio), int(self.raw_cam_img_size[1]//self.ratio)))
        sn_np = np.transpose(sn_np, (2, 0, 1))[:1, :, :]  # Shape: (3, H, W)
        sn_np[sn_np > 0] = 1 
        # if sn.shape[0] == 3:
        #     sn = sn.transpose(1, 2, 0)  # Convert to (H, W, 3) if needed
        # sn = torch.from_numpy(sn).permute(2, 0, 1).float()  # Shape: (3, H, W)

        # Load trajectory and generate footprint mask
        # traj = np.load(sample["traj"])  # Shape: (N, 2) with x, y coordinates
        # fp = torch.from_numpy(fp).long()  # Shape: (H, W)
        fp_np = thermal_read(S["footprint"])[...,0]
        fp_np = resize_rgb_image(fp_np,
                            (self.raw_cam_img_size[0]//self.ratio,
                             self.raw_cam_img_size[1]//self.ratio))
        fp_np_3ch = np.repeat(fp_np[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        # fp_np = np.transpose(fp_np_3ch, (2, 0, 1))
        fp_np = np.transpose(fp_np_3ch, (2, 0, 1))[:1, :, :]
        # fp = (fp>0).astype(np.uint8)[None,...]
        fp_np[fp_np > 0] = 1
        # Ground truths dictionary

        gts = {
            "sn": sn,  # Surface normals
            "fp": fp_np,   # Footprint mask
        }

        fname = S["fname"]
        return rgbd, gts, fname