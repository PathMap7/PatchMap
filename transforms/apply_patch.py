import torch
from transforms.my_random_affine import MyRandomAffine, F


class ApplyPatch(torch.nn.Module):

    def __init__(self, patch, translation_range=(.2, .2), rotation_range=45,
                 scale_range=(0.5, 1), patch_size=50, translations=[]):
        super().__init__()
        self.patch_size = patch_size
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range

        self._transforms = None
        self._patch = None
        self._input_shape = None
        self._mask = None

        self.set_transforms(translation_range, rotation_range, scale_range)
        self.set_patch(patch)

    @property
    def mask(self):
        return self._mask

    @property
    def transforms(self):
        return self._transforms

    def set_patch(self, patch):
        self._patch = patch
        self._input_shape = self._patch.shape
        self._mask = self._generate_mask()

    def _generate_mask(self):
        ## here mask is determined on the center of the image
        mask = torch.ones(self._input_shape)
        upp_l_x = self._input_shape[2] // 2 - self.patch_size // 2
        upp_l_y = self._input_shape[1] // 2 - self.patch_size // 2
        bott_r_x = self._input_shape[2] // 2 + self.patch_size // 2
        bott_r_y = self._input_shape[1] // 2 + self.patch_size // 2
        mask[:, :upp_l_x, :] = 0
        mask[:, :, :upp_l_y] = 0
        mask[:, bott_r_x:, :] = 0
        mask[:, :, bott_r_y:] = 0

        return mask

    def set_transforms(self, translation_range, rotation_range,
                       scale_range):
        self._transforms = MyRandomAffine(
            rotation_range, translation_range, scale_range)

    def forward(self, img, translate=None):
        ## translate - distance from center, we can find it from bounding box ####
        ## random translate from center        
        patch, mask = self.transforms(self._patch, self._mask)
        n = patch.shape[0]
        patchmask = torch.cat([patch, mask], dim=0)
        patchmask = F.affine(patchmask, angle=0.0, scale=1.0, shear=0.0, translate=[translate[0], translate[1]])
        patch, mask = patchmask[:n], patchmask[n:]
        inv_mask = torch.zeros_like(mask)
        inv_mask[mask == 0] = 1
        return img * inv_mask + patch * mask
