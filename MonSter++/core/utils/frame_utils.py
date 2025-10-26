import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


import OpenEXR
import pyexr
import Imath
import math
import h5py
import torch

def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def read_exr(file_name):
    exr_file = OpenEXR.InputFile(file_name)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = header['channels'].keys()

    # 选择第一个通道（大多数深度图只存一个通道）
    first_channel = list(channels)[0]
    depth = np.frombuffer(exr_file.channel(first_channel, FLOAT), dtype=np.float32)

    return depth.reshape((size[1], size[0]))

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())



def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid

def readDispDrivingStereoFull(filename, is_full=False):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 128.0
    valid = disp > 0.0
    return disp, valid

def readDispDrivingStereo_half(filename, is_full=False):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid


def readDispCREStereo(filename):
    disp = np.array(Image.open(filename))
    return disp.astype(np.float32) / 32.

def readDispInStereo2K(filename):
    disp = np.array(Image.open(filename))
    disp = disp.astype(np.float32) / 100.
    valid = disp > 0.0
    return disp, valid

def readDispVKITTI2(filename):
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = (depth / 100).astype(np.float32)
    valid = (depth > 0) & (depth < 655)
    
    focal_length = 725.0087
    baseline = 0.532725
    disp = baseline * focal_length / depth
    disp[~valid] = 0.0

    return disp, valid

# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2**6) + d_b / (2**14))[..., 0]
    mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid

# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open('/'.join(file_name.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid

# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
# def readDispTartanAir(file_name):
#     try:
#         depth = np.load(file_name)
#     except:
#         print('file name', file_name)
#         depth = None
#     if depth is not None:
#         disp = 80.0 / depth
#         valid = disp > 0
#     else:
#         disp = None
#         valid = None
#     return disp, valid

def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid


def readDispMiddlebury(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png':
        disp = np.array(Image.open(file_name)).astype(np.float32)
        valid = disp > 0.0
        return disp, valid
    elif basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    else:
        disp = readPFM(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        valid = disp > 0.0
        return disp, valid    

def readFoundationStereo(filename, scale=1000):
    disp = imageio.imread(filename).astype(float)
    disp = disp[...,0]*255*255 + disp[...,1]*255 + disp[...,2]
    disp = disp / float(scale)
    valid = disp > 0.0
    return disp, valid

def readDispIRS(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.squeeze(hdr)
	return hdr

def readDispBooster(file_name):
    disp = np.load(file_name)
    valid = disp > 0
    return disp, valid

def readDisp3DKenBurns(file_name):
    depth = read_exr(file_name).astype(np.float32)
    meta_file_name = file_name.replace('-depth', '')[:-7]+'-meta.json'
    fltFov = json.loads(open(meta_file_name, 'r').read())['fltFov']
    fltFocal = 0.5 * 512 * math.tan(math.radians(90.0) - (0.5 * math.radians(fltFov)))
    fltBaseline = 40.0
    disp = (fltFocal * fltBaseline) / depth
    valid = disp > 0
    return disp, valid

def readDispVA(file_name):
    depth = readPFM(file_name).astype(np.float32)
    focal_length = 320
    baseline = 0.131202
    # disp = (focal_length * baseline) / depth
    valid_mask = depth > 1e-6
    disp = np.zeros_like(depth)
    disp[valid_mask] = (focal_length * baseline) / depth[valid_mask]
    valid = disp > 0
    return disp, valid

def readDispSimSIN(file_name):
    depth = np.load(file_name).astype(np.float32)
    focal_length = 256
    baseline = 0.131202
    # disp = (focal_length * baseline) / depth
    valid_mask = depth > 1e-6
    disp = np.zeros_like(depth)
    disp[valid_mask] = (focal_length * baseline) / depth[valid_mask]
    valid = disp > 0
    return disp, valid

def readDispUnrealStereo4K(file_name):
    disp = np.load(file_name).astype(np.float32)
    valid = (disp > 0) & (disp < np.inf)
    return disp, valid

def readDispSpring(file_name):
    disp = np.asarray(h5py.File(file_name)['disparity']).astype(np.float32)
    disp = disp[::2, ::2]
    disp[np.isnan(disp)] = 0 # make invalid values as +inf
    #disp[disp==0.0] = np.inf # make invalid values as +inf
    valid = disp > 0
    return disp, valid

def readDispStereoBlur(file_name):
    disp = pyexr.open(file_name).get().squeeze()
    valid = (disp > 0) & (disp < np.inf )
    disp[~valid] = 0
    # print(f'disp max: {disp[valid].max()}, disp min: {disp[valid].min()}')
    return disp.astype(np.float32), valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg' or ext == '.tif' or ext == '.tiff':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []