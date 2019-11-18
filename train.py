import matplotlib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
from skimage.transform import resize as imresize
import scipy.ndimage
from utils import now
from travelgan import TravelGAN
from loader import Loader
from PIL import Image

def get_data_args(args):
    train, label, args.channels, args.imdim = many2one_dataset(args.datadirs, D=int(1.25 * args.downsampledim))
    return args, train, label

def many2one_dataset(datadirs,delimeter= '_', D=128):
    """
    ##TODO replace this temporary function with a data loader.
    datadirs: directories of the data, the last directory used as label, others
    used as training dataset.
    """
    dataset_n = len(datadirs)
    if dataset_n<2:
        raise ValueError("At least 2 image directories need to be provided.")
    batches = []
    markers = []
    name_order = []
    dimensions = []
    offset = []
    for datadir in datadirs:
        b,f_n,D= preprocess_img(datadir,D=D,compress_to_gray = True)
        dimensions.append(D)
        marker = np.asarray([delimeter.join(os.path.basename(x).split(delimeter)[:-1]) for x in f_n])
        name_order.append(np.argsort(marker))
        batches.append(b[name_order])
        markers.append(marker[name_order])
        offset.append(0)
    inter_fns = set(markers[0])
    n = 0
    for fn in markers[1:]:
        inter_fns = inter_fns.intersection(set(fn))
        n = max(n,len(batches))
        
    match = True
    for marker in markers:
        if inter_fns != marker:
            match = False
    if not match:
        print("Warning, some image is not paired and will be ignored.")
        train = []
        label = []
        for img_idx,img in batches[0]:
            curr_img = [img]
            fit = True
            if markers[0][img_idx] not in inter_fns:
                continue
            for b_idx,b in enumerate(batches[1:]):
                while markers[b_idx][offset[b_idx]] < markers[0][img_idx]:
                    offset[b_idx]+=1
                if markers[b_idx][offset[b_idx]] != markers[0][img_idx]:
                    fit = False
                    break
                curr_img.append[np.reshape(b[offset[b_idx]],shape = (D,D))]
            if fit:
                train.append(np.stack(curr_img[:-1],axis = 2))
                label.append(np.reshape(curr_img[-1],shape = (D,D,1)))
        train = np.stack(train,axis = 0)
        label = np.stack(label,axis = 0)
    else:
        label = batches[:-1]
        train = batches[1:]
        train = [np.stack(x,axis =0 ) for x in train]
        train = [np.reshape(x,(x.shape[0],x.shape[1],x.shape[2])) for x in train]
        train = np.stack(train,axis=3)
    channels = train.shape[3]
    return train, label, channels, int(.8 * D)

def preprocess_img(datadir,D = 512,compress_to_gray = True):
    fns = sorted(glob.glob('{}/*'.format(datadir)))
    fns = [fn for fn in fns if any(['tif' in fn.lower(),'tiff' in fn.lower(), 'png' in fn.lower(), 'jpeg' in fn.lower(), 'jpg' in fn.lower()])]
    if compress_to_gray:
        compress = lambda x: np.sum(np.reshape(x,(x.shape[0],x.shape[1],1)),axis=2,keepdims = True) if len(x.shape)==2 else np.sum(x,axis=2,keepdims = True)
    else:
        compress = lambda x: x

    imgs = [[compress(np.asarray(Image.open(f).resize((D,D))))] for f in fns]
    print("compress image size")
    return np.asarray(imgs), np.asarray(fns), int(.8 * D)

def randomize_image(img, enlarge_size=286, output_size=256):
    img = imresize(img, [enlarge_size, enlarge_size])
    h1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size - output_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size - output_size)))
    img = img[h1:h1 + output_size, w1:w1 + output_size]
    if np.random.random() > .5:
        img = np.fliplr(img)

    return img

def randomcrop(imgs, cropsize):
    imgsout = np.zeros((imgs.shape[0], cropsize, cropsize, imgs.shape[3]))
    for i in range(imgs.shape[0]):
        img = imgs[i]
        h1 = int(np.ceil(np.random.uniform(1e-2, img.shape[1] - cropsize)))
        w1 = int(np.ceil(np.random.uniform(1e-2, img.shape[1] - cropsize)))
        img = img[h1:h1 + cropsize, w1:w1 + cropsize]
        if np.random.random() > .5:
            img = np.fliplr(img)
        imgsout[i] = img
    return imgsout

def central_crop(imgs,cropratio):
    width = imgs.shape[1]
    cropsize = int(np.floor(width*cropratio))
    imgsout = np.zeros((imgs.shape[0], cropsize, cropsize, imgs.shape[3]))
    for i in range(imgs.shape[0]):
        img = imgs[i]
        h1 = int(np.ceil(width*cropratio/2.0))
        w1 = int(np.ceil(width*cropratio/2.0))
        img = img[h1:h1 + cropsize, w1:w1 + cropsize]
        imgsout[i] = img
    return imgsout


def parse_args():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument('--savefolder', type=str)

    # data params
    parser.add_argument('--downsampledim', type=int, default=128)
    parser.add_argument('--datadirs', type=str, default='',
                        help = "Data directories divided by comma.")
    parser.add_argument('--paired_data', type=int, default=False)

    # model params
    parser.add_argument('--nfilt', type=int, default=64)
    parser.add_argument('--lambda_adversary', type=float, default=1)

    # siamese params
    parser.add_argument('--lambda_siamese', type=float, default=10)
    parser.add_argument('--siamese_latentdim', type=int, default=1000)

    # training params
    parser.add_argument('--training_steps', type=int, default=200000)
    parser.add_argument('--training_steps_decayafter', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_frac', type=float, default=1)
    parser.add_argument('--restore_folder', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=.0002)

    args = parser.parse_args()

    args.modelname = 'TraVeLGAN'
    args.model = TravelGAN
    args.datadirs = args.datadirs.split(',')

    return args

args = parse_args()

### Local test
args.savefolder = "/media/heavens/LaCie/Murphy_data/out"
args.datadirs = "/media/heavens/LaCie/Murphy_data/images/blue,\
                /media/heavens/LaCie/Murphy_data/images/green,\
                /media/heavens/LaCie/Murphy_data/images/yellow,\
                /media/heavens/LaCie/Murphy_data/images/red".split(',')
args.downsampledim = 256
args.paired_data = True
if not os.path.exists(args.savefolder): 
    os.mkdir(args.savefolder)
if not args.restore_folder:
    with open(os.path.join(args.savefolder, 'args.txt'), 'w+') as f:
        for arg in vars(args):
            argstring = "{}: {}\n".format(arg, vars(args)[arg])
            f.write(argstring)
            print(argstring[:-1])

if not os.path.exists("{}/output".format(args.savefolder)): os.mkdir("{}/output".format(args.savefolder))
###
args,train,label = get_data_args(args)
load1 = Loader(train, labels=np.arange((train.shape[0])), shuffle=False)
load2 = Loader(label, labels=np.arange((label.shape[0])), shuffle=False)

print("Domain 1 shape: {}".format(train.shape))
print("Domain 2 shape: {}".format(label.shape))
model = args.model(args, x1=train, x2=label, name=args.modelname)


plt.ioff()
fig = plt.figure(figsize=(4, 10))
np.set_printoptions(precision=3)
decay = model.args.learning_rate / (args.training_steps - args.training_steps_decayafter)
crop_ratio = 0.8
for i in range(1, args.training_steps):

    if i % 10 == 0: print("Iter {} ({})".format(i, now()))
    model.train()

    if i >= args.training_steps_decayafter:
        model.args.learning_rate -= decay

    if i and (i == 50 or i % 500 == 0):
        model.save(folder=args.savefolder)

        xb1inds = np.random.choice(train.shape[0], replace=False, size=[10])
        if args.paired_data:
            xb2inds = xb1inds
        else:
            xb2inds = np.random.choice(label.shape[0], replace=False, size=[10])
        testb1 = train[xb1inds]
        testb2 = label[xb2inds]

#        testb1 = randomcrop(testb1, args.imdim)
#        testb2 = randomcrop(testb2, args.imdim)
        testb1 = central_crop(testb1, crop_ratio)
        testb2 = central_crop(testb2, crop_ratio)
        

        Gb1 = model.get_layer(testb1, testb2, name='Gb1')
        Gb2 = model.get_layer(testb1, testb2, name='Gb2')

        # back to [0,1] for imshow
        testb1 = (testb1 + 1) / 2
        testb2 = (testb2 + 1) / 2
        Gb1 = (Gb1 + 1) / 2
        Gb2 = (Gb2 + 1) / 2

        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb1[ii])
            ax2.imshow(Gb2[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b1_to_b2.png'.format(args.savefolder), dpi=500)

        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb2[ii])
            ax2.imshow(Gb1[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b2_to_b1.png'.format(args.savefolder), dpi=500)


        xb1inds = np.random.choice(train.shape[0], replace=False, size=[args.batch_size])
        xb2inds = np.random.choice(label.shape[0], replace=False, size=[args.batch_size])
        testb1 = train[xb1inds]
        testb2 = label[xb2inds]

#        testb1 = randomcrop(testb1, args.imdim)
#        testb2 = randomcrop(testb2, args.imdim)

        testb1 = central_crop(testb1, crop_ratio)
        testb2 = central_crop(testb2, crop_ratio)

        print(model.get_loss_names())
        lstring = model.get_loss(testb1, testb2)
        print("{} ({}): {}".format(i, now(), lstring))











