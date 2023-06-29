import os 
import nibabel as nib
import numpy as np


data_path = r'Y:\users\ealvarezandres\Preprocess_data_ni'
patients = os.listdir(data_path)
save_path = r'./data/'

def clip(data, center, width):
    return np.clip(data, center - width/2, center + width/2)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def create_bounding_box(mask):
    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))

    x_min, x_max = np.where(x)[0][[0, -1]]
    y_min, y_max = np.where(y)[0][[0, -1]]
    z_min, z_max = np.where(z)[0][[0, -1]]

    return x_min, x_max, y_min, y_max, z_min, z_max



def pad_box(box, shape):
    x_min, x_max, y_min, y_max, z_min, z_max = box
    x1 =  x_min - (shape[0] - (x_max - x_min))//2
    x2 = x_max + (shape[0] - (x_max - x_min))//2
    y1 = y_min - (shape[1] - (y_max - y_min))//2
    y2 = y_max + (shape[1] - (y_max - y_min))//2
    return [x1, x2, y1, y2, z_min, z_max]


for patient in patients:
    print(patient)
    path2 = os.path.join(data_path, patient, 'acquisition_0')
    if patient!='other' and os.path.exists(os.path.join(path2, patient+'_'+ 'CT.nii')) and os.path.exists(os.path.join(path2, patient+'_'+ 'MR1_T1.nii')) and os.path.exists(os.path.join(path2, patient+'_'+ 'MR1_T1_bet_mask.nii')):
        ct = nib.load(os.path.join(path2, patient+'_'+ 'CT.nii'))
        mri = nib.load(os.path.join(path2, patient+'_'+ 'MR1_T1.nii'))
        brain_mask = nib.load(os.path.join(path2, patient+'_'+ 'MR1_T1_bet_mask.nii'))
    else:
        print("patient {} issue".format(patient))
        continue
    
    ct_data = ct.get_fdata()
    mri_data = mri.get_fdata()
    brain_mask_data = brain_mask.get_fdata()
    
    bounding_box = create_bounding_box(brain_mask_data)
    padded_bounding_box = pad_box(bounding_box, (224,224))
    if padded_bounding_box[1] - padded_bounding_box[0]==223:
        padded_bounding_box[0] = padded_bounding_box[0] - 1
    if padded_bounding_box[3] - padded_bounding_box[2] == 223:
        padded_bounding_box[2] = padded_bounding_box[2] - 1
    cropped_ct = clip(ct_data[padded_bounding_box[0]:padded_bounding_box[1], padded_bounding_box[2]:padded_bounding_box[3], padded_bounding_box[4]:padded_bounding_box[5]], 50,100)
    cropped_mri = mri_data[padded_bounding_box[0]:padded_bounding_box[1], padded_bounding_box[2]:padded_bounding_box[3], padded_bounding_box[4]:padded_bounding_box[5]]

    processed_ct, processed_mr = normalize(cropped_ct), normalize(cropped_mri)
    if processed_ct.shape[0] != 224 or processed_ct.shape[1] != 224 or processed_mr.shape[0]!=224 or processed_mr.shape[1]!= 224:
        print(processed_ct.shape, processed_mr.shape)
        print("Pb shape for patient {}".format(patient))
    os.makedirs(os.path.join(save_path, patient), exist_ok=True)
    nib.save(nib.Nifti1Image(processed_ct, ct.affine), os.path.join(save_path, patient,  patient+'_'+ 'CT.nii.gz'))
    nib.save(nib.Nifti1Image(processed_mr, mri.affine), os.path.join(save_path, patient,  patient+'_'+ 'MR1_T1.nii.gz'))
    print("patient {} done".format(patient))