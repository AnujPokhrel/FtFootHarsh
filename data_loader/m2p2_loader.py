import os
import numpy as np
import torch
import torch.utils.data as data
import random
import natsort
import pdb
from common.utils_loader import rgb_read, resize_rgb_image, sn_image_from_npy, crop_image, thermal_read, depth_read, resize_sn_image

__all__ = ['M2P2']

class M2P2(data.Dataset):

    def __init__(self, cfg, mode="test"):
        self.cfg = cfg
        self.mode           = mode                        # "train"/"valid"/"test"
        self.data_root      = cfg.data_root               # root of your dataset
        self.raw_cam_img_size = tuple(cfg.raw_cam_img_size)     # e.g. [480, 640]
        self.ratio          = cfg.ratio                   # any down‑sampling factor
        self.load_interval  = cfg.get("load_interval", 1)
        self.num_samples    = getattr(cfg, f"num_{mode}_samples", -1)

        # Define base directories for each modality 
        self.img_dir = os.path.join(self.data_root, mode, "thermal")
        self.depth_dir = os.path.join(self.data_root, mode, "depth")
        self.sn_dir  = os.path.join(self.data_root, mode, "surface_normal")
        self.fp_dir  = os.path.join(self.data_root, mode, "footprint")
        # pdb.set_trace() 
        # files    = natsort.natsorted(os.listdir(self.img_dir))
        samples  = []
        depth_chunks = [d for d in os.listdir(self.depth_dir) if os.path.isdir(os.path.join(self.depth_dir, d))]
        depth_chunks = natsort.natsorted(depth_chunks)

        # for fn in files[::self.load_interval]:
        #     base = os.path.splitext(fn)[0]
        #     entry = {
        #         "therm":      os.path.join(img_dir, fn),
        #         "depth":      os.path.join(depth_dir, base + ".png"),
        #         "sn":         os.path.join(sn_dir,  base + ".npy"),
        #         "footprint":  os.path.join(fp_dir,  base + ".png"),
        #         "fname":      base
        #     }
        #     samples.append(entry)

        for chunk in depth_chunks:
            # extract chunk identifier
            chunk_id = chunk.split("_")[-2] + "_" + chunk.split("_")[-1]
            timestamp = "_".join(chunk.split("_")[1:-2])

            depth_chunk_path = os.path.join(self.depth_dir, chunk)
            depth_files = natsort.natsorted([f for f in os.listdir(depth_chunk_path) if f.endswith(".png")])

            thermal_chunk_path = os.path.join(self.thermal_dir, f"thermal_{timestamp}_{chunk_id}_processed")
            sn_chunk_path = os.path.join(self.sn_dir, f"BL_{timestamp}_{chunk_id}")
            fp_chunk_path = os.path.join(self.fp_dir, f"BL_{timestamp}_{chunk_id}_footprint_mask")

            for i, depth_file in enumerate(depth_files[::self.load_interval]):
                # Extract file index (e.g., "0" from "0.png")
                file_idx = os.path.splitext(depth_file)[0]

                # Construct corresponding file paths
                depth_path = os.path.join(depth_chunk_path, f"{file_idx}.png")
                thermal_path = os.path.join(thermal_chunk_path, f"{file_idx}.png")  # Adjust if thermal naming differs
                sn_path = os.path.join(sn_chunk_path, f"{file_idx}_sparse.npy")
                fp_path = os.path.join(fp_chunk_path, f"{file_idx:06d}.png")  # Footprint uses 6-digit padding

                # Check if all files exist
                if (os.path.exists(depth_path) and os.path.exists(thermal_path) and 
                    os.path.exists(sn_path) and os.path.exists(fp_path)):
                    entry = {
                        "therm": thermal_path,
                        "depth": depth_path,
                        "sn": sn_path,
                        "footprint": fp_path,
                        "fname": f"{timestamp}_{chunk_id}_{file_idx}"
                    }
                    samples.append(entry)


        if self.num_samples>0: 
            samples = samples[:self.num_samples]
        if mode=="train":  
            random.shuffle(samples)
        self.samples = samples
        print(f"############Here is the Samples list: {self.samples}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        S = self.samples[idx]

        therm = thermal_read(S["therm"])
        therm_3ch = np.repeat(therm[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        # print(therm_3ch.shape)
        therm_3ch = crop_image(therm_3ch, self.raw_cam_img_size)
        therm_3ch = resize_rgb_image(therm_3ch, 
                                (self.raw_cam_img_size[0]//self.ratio,
                                self.raw_cam_img_size[1]//self.ratio))

        # Repeat thermal channel to mimic RGB (3 channels)

        depth = depth_read(S["depth"])
        depth = crop_image(depth, self.raw_cam_img_size)
        depth = resize_rgb_image(depth, 
                                (self.raw_cam_img_size[0]//self.ratio,
                                self.raw_cam_img_size[1]//self.ratio))

        # therm_3ch = np.repeat(therm_3ch[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        rgbd = np.concatenate((therm_3ch, depth[:, :, np.newaxis]), axis=2)  # Shape: (H, W, 4)
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1)  # Shape: (4, H, W)
        # rgbd_tensor = torch.from_numpy(rgbd).cuda().float()

        # Load surface normals (assuming 3-channel NumPy array)
        sn = np.load(S["sn"])  # Shape: (H, W, 3) or (3, H, W)
        sn_np = sn_image_from_npy(sn, self.raw_cam_img_size, px=3)
        sn_np = resize_sn_image(sn_np, (int(self.raw_cam_img_size[0]//self.ratio), int(self.raw_cam_img_size[1]//self.ratio)))
        sn_np = np.transpose(sn_np, (2, 0, 1))[:1, :, :]  # Shape: (3, H, W)
        sn_np[sn_np > 0] = 1 
        # sn_tensor = torch.from_numpy(sn_np).cuda().float()

        fp_np = thermal_read(S["footprint"])[...,0]
        fp_np = resize_rgb_image(fp_np,
                            (self.raw_cam_img_size[0]//self.ratio,
                             self.raw_cam_img_size[1]//self.ratio))
        fp_np_3ch = np.repeat(fp_np[:, :, np.newaxis], 3, axis=2)  # Shape: (H, W, 3)
        fp_np = np.transpose(fp_np_3ch, (2, 0, 1))[:1, :, :]
        fp_np[fp_np > 0] = 1
        # fp_tensor = torch.from_numpy(fp_np).cuda().float()
        # Ground truths dictionary

        gts = {
            "sn": sn_np,  # Surface normals
            "fp": fp_np,   # Footprint mask
        }
        # gts = {
        #     "sn": sn_tensor,  # Surface normals
        #     "fp": fp_tensor,   # Footprint mask
        # }
        fname = S["fname"]
        return rgbd, gts, fname