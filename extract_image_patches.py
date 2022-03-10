import numpy

def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = numpy.meshgrid( numpy.arange(shape[1]),numpy.arange(shape[0]))
    l, t = numpy.meshgrid(
        numpy.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        numpy.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = numpy.tile( px[None,:,:], (t.size,1,1)) + numpy.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = numpy.tile( py[None,:,:], (t.size,1,1)) + numpy.tile( t[:,None,None], (1,shape[0],shape[1]))
    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t,l)

def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = numpy.max( origin[1] ) + patch_width
    img_height   = numpy.max( origin[0] ) + patch_height

    out = numpy.zeros( (img_height,img_width) )
    wgt = numpy.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/numpy.maximum( wgt, epsilon ), wgt

if __name__ == '__main__':
    import cv2
    import time
    import matplotlib.pyplot as plt
    from PIL import Image
    import os 
    import glob 
    import numpy as np
    
    patch_sz = (300,300)
    
    # stride (2-element arraylike): vertical and horizontal strides
#    patch_stride = (100,50)   # this creates 50 ppi (10x5)
    patch_stride = (75,50) # 65ppi (13x5)   

#    src = '/media/adjeroh/10TBHD/DATABASES/Rail_Surface_Database/rail_images_trainval/train'
#    dst = '/home/adjeroh/extreme_scales/data/rail_images_300x300_65ppi/train' 
    
    src = '/media/adjeroh/10TBHD/DATABASES/Rail_Surface_Database/rail_images_trainval/val'
    dst = '/home/adjeroh/extreme_scales/data/rail_images_300x300_65ppi/val' 
    
    dirlist = os.listdir(src)
    
    for dname in dirlist:
        print('processing {}... '.format(dname))
        filepathlist = glob.glob(os.path.join(src, dname, '*.bmp'))
        i = 1
        for fpath in filepathlist:
            print('{}/{} {}'.format(i,len(filepathlist), os.path.basename(fpath)))
            i = i + 1
            img = cv2.imread(fpath)[:,:,::-1]
            img = cv2.resize(img, (500, 1200)) # (width, height) 
            print('img shape: ({},{})'.format(img.shape[0],img.shape[1]))
            patches, origin = extract_grayscale_patches( img[:,:,2], patch_sz, stride=patch_stride )
            print('{} patches created. '.format(patches.shape[0]))
            filename = os.path.basename(fpath).split('.')[0]
            if not os.path.isdir(os.path.join(dst,dname)):
                os.makedirs(os.path.join(dst,dname))
            for p in range(patches.shape[0]):
                im = Image.fromarray(np.uint8(patches[p]))
                top = origin[0][p]
                left = origin[1][p]
                im.save(os.path.join(dst, dname, filename + '_' + str(top) + '_' + str(left) + '.bmp'))
    
