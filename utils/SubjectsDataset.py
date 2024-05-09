import os
import torchio as tio


class SubjectsDataset(tio.SubjectsDataset):

    
    def __init__(self, root):
        _, _, img_files = next(os.walk(os.path.join(root, 'image')))
        _, _, mask_files = next(os.walk(os.path.join(root, 'mask')))
        
        if set(img_files) != set(mask_files):
            raise ValueError('Image and mask files do not match')

        self._subjects = [
            tio.Subject(
                image=tio.ScalarImage(os.path.join(root, 'image', f)),
                mask=tio.LabelMap(os.path.join(root, 'mask', f))
            )
            for f in sorted(img_files)
        ]
        self._transform = None
        self.load_getitem = True
