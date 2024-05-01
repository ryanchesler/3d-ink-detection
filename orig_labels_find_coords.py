import json
with open("/home/ryanc/precise_labels.json") as f:
    boxes = json.load(f)
import cv2
from tqdm import tqdm

# %%
import pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pathlib
import h5py

# %%

class Ppm():

    def __init__(self):
        self.data = None
        self.ijks = None
        self.normals = None
        self.ijk_interpolator = None
        self.normal_interpolator = None
        self.data_header = None
        self.valid = False
        self.error = "no error message set"

    def createErrorPpm(err):
        ppm = Ppm()
        ppm.error = err
        return ppm

    no_data = (0.,0.,0.)

    # lijk (layer ijk) is in layer's global coordinates
    def layerIjksToScrollIjks(self, lijks):
        print("litsi")
        if self.data is None:
            print("litsi no data")
            return lijks
        '''
        li,lj,lk = lijk
        if li < 0 or lj < 0:
            return Ppm.no_data
        if li >= self.width:
            return Ppm.no_data
        if lj >= self.height:
            return Ppm.no_data
        '''
        # sijk = np.zeros((lijk.shape), dtype=lijk.dtype)
        # sijks = self.ijk_interpolator(lijks[:,0:2])
        ijs = lijks[:,(2,0)]
        ks = lijks[:,1,np.newaxis]
        sijks = self.ijk_interpolator(ijs)
        norms = self.normal_interpolator(ijs)
        print(lijks.shape, sijks.shape, norms.shape, ks.shape)
        sijks += norms*(ks-32)
        return sijks


    def loadData(self):
        if self.data is not None:
            return
        print("reading data from %s for %s"%(str(self.path), self.name))

        fstr = str(self.path)
        print("reading ppm data for", self.path)
        if not self.path.exists():
            err="ppm file %s does not exist"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            fd = self.path.open("rb")
        except Exception as e:
            err="Failed to open ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            bdata = fd.read()
        except Exception as e:
            err="Failed to read ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        index = bdata.find(b'<>\n')
        if index < 0:
            err="Ppm file %s does not have a header"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        bdata = bdata[index+3:]
        lbd = len(bdata)
        height = self.height
        width = self.width
        le = height*width*8*6
        if lbd != le:
            err="Ppm file %s expected %d bytes of data, got %d"%(fstr, le, lbd)
            print(err)
            return Ppm.createErrorPpm(err)

        raw = np.frombuffer(bdata, dtype=np.float64)
        self.data = np.reshape(raw, (height,width,6))
        self.ijks = self.data[:,:,:3]
        self.normals = self.data[:,:,3:]
        print(self.ijks.shape, self.normals.shape)
        # print(self.ijks[0,0,:],self.normals[0,0,:])
        # print(self.ijks[3000,3000,:],self.normals[3000,3000,:])
        ii = np.arange(height)
        jj = np.arange(width)
        self.ijk_interpolator = RegularGridInterpolator((ii, jj), self.ijks, fill_value=0., bounds_error=False)
        self.normal_interpolator = RegularGridInterpolator((ii, jj), self.normals, fill_value=0., bounds_error=False)



    # reads and loads the header of the ppm file
    def loadPpm(filename):
        fstr = str(filename)
        print("reading ppm header for", filename)
        if not filename.exists():
            err="ppm file %s does not exist"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            fd = filename.open("rb")
        except Exception as e:
            err="Failed to open ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            bstr = fd.read(200)
        except Exception as e:
            err="Failed to read ppm file %s: %s"%(fstr, e)
            print(err)
            return Ppm.createErrorPpm(err)

        index = bstr.find(b'<>\n')
        if index < 0:
            err="Ppm file %s does not have a header"%fstr
            print(err)
            return Ppm.createErrorPpm(err)

        hstr = bstr[:index+3].decode('utf-8')
        lines = hstr.split('\n')
        hdict = {}
        for line in lines:
            words = line.split()
            if len(words) != 2:
                continue
            name = words[0]
            value = words[1]
            if name[-1] != ':':
                continue
            name = name[:-1]
            hdict[name] = value
        for name in ["width", "height"]:
            if name not in hdict:
                err="Ppm file %s missing \"%s\" in header"%(fstr, name)
                print(err)
                return Ppm.createErrorPpm(err)

        try:
            width = int(hdict["width"])
        except Exception as e:
            err="Ppm file %s could not parse width value \"%s\" in header"%(fstr, hdict["width"])
            print(err)
            return Ppm.createErrorPpm(err)

        try:
            height = int(hdict["height"])
        except Exception as e:
            err="Ppm file %s could not parse height value \"%s\" in header"%(fstr, hdict["height"])
            print(err)
            return Ppm.createErrorPpm(err)

        expected = {
                "dim": "6",
                "ordered": "true",
                "type": "double",
                "version": "1",
                }

        for name, value in expected.items():
            if name not in hdict:
                err = "Ppm file %s missing \"%s\" from header"%(fstr, name)
                print(err)
                return Ppm.createErrorPpm(err)
            if hdict[name] != expected[name]:
                err = "Ppm file %s expected value of \"%s\" for \"%s\" in header; got %s"%(fstr, expected[name], name, hdict[name])
                print(err)
                return Ppm.createErrorPpm(err)
    
        ppm = Ppm()
        ppm.valid = True
        ppm.height = height
        ppm.width = width
        ppm.path = filename
        ppm.name = filename.stem
        print("created ppm %s width %d height %d"%(ppm.name, ppm.width, ppm.height))
        return ppm
    
import os
pos_coords = []
neg_coords = []
for file, regions in boxes["_via_img_metadata"].items():
    labeled_boxes = []
    for region in regions["regions"]:
        region = region["shape_attributes"]
        labeled_boxes.append([region["x"], region["y"], region["x"]+region["width"], region["y"]+region["height"]])
    segment = regions["filename"].split("_")[0]
    if segment == "20230904135535":
        continue
    predictions = cv2.imread(f"/home/ryanc/kaggle/labels/{segment}_inklabels.png", 0)
    ppm = Ppm.loadPpm(pathlib.Path(f"/data/scroll_data/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/{segment}/{segment}.ppm"))
    ppm.loadData()
    for box in labeled_boxes:
        ys = list(range(box[1], box[3]))
        xs = list(range(box[0], box[2]))
        for y in tqdm(ys):
            for x in xs:
                if predictions[y, x] == 255:
                    pos_coords.append(ppm.ijks[y, x])
                if predictions[y, x] == 0:
                    neg_coords.append(ppm.ijks[y, x])
coords = np.array(pos_coords)
print(coords.shape)
np.save("train_pos_coords.npy", coords)

coords = np.array(neg_coords)
print(coords.shape)
np.save("train_neg_coords.npy", coords)
        