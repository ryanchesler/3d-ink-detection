import cv2
from tqdm import tqdm
import pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py

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
    
hdf_path = '/data/all_labels_train.hdf5'
with h5py.File(hdf_path, 'w') as f:
    dset = f.create_dataset("scan_predictions", shape=(7888, 8096, 14376), dtype=np.float32, chunks=(32, 32, 32), compression="gzip")
    dset = f.create_dataset("scan_counts", shape=(7888, 8096, 14376), dtype=np.uint8, chunks=(32, 32, 32), compression="gzip")
import os
last_segment = None
for file in sorted(os.listdir("/home/ryanc/kaggle/all_labels"), reverse=True):
    try:
        segment = "_".join(file.split("_")[:-1])
        print(segment)
        if segment == "20230904135535":
            continue
        if last_segment is not None:
            if last_segment[:-1] == segment[:-1]:
                print("skipping because older version", segment)
                continue
        last_segment = segment
        ppm = Ppm.loadPpm(pathlib.Path(f"/data/scroll_data/dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/{segment}/{segment}.ppm"))
        ppm.loadData()
        predictions = cv2.imread(f"/home/ryanc/kaggle/all_labels/{segment}_inklabels.png", 0)
        print(predictions.max())
        labeled_boxes = [[0, 0, -1, -1]]
        group_size = 128
        with h5py.File(hdf_path, 'r+') as f:
            for box in labeled_boxes:
                sub_predictions = predictions[box[1]:box[3], box[0]:box[2]]
                ijks = ppm.ijks[box[1]:box[3], box[0]:box[2]]
                normals = ppm.normals[box[1]:box[3], box[0]:box[2]]
                pbar = tqdm(range(0, ijks.shape[0], group_size))
                for y in pbar:
                    for x in range(0, ijks.shape[1], group_size):
                        if predictions[y:y+group_size, x:x+group_size].max() == 0:
                            continue
                        lower_scroll_cords = ijks[y:y+group_size, x:x+group_size] + (((32+4)-(65//2)) * normals[y:y+group_size, x:x+group_size])
                        upper_scroll_cords = ijks[y:y+group_size, x:x+group_size] + (((32-4)-(65//2)) * normals[y:y+group_size, x:x+group_size])
                        min_coords = np.floor(np.stack([lower_scroll_cords, upper_scroll_cords], axis = 0).min(0)).astype(int)
                        max_coords = np.ceil(np.stack([lower_scroll_cords, upper_scroll_cords], axis = 0).max(0)).astype(int)
                        if upper_scroll_cords.max() > 0. and lower_scroll_cords.min() > 0.:
                            sub_volume = f["scan_predictions"][min_coords[:, :, 1].min():max_coords[:, :, 1].max(),
                                        min_coords[:, :, 0].min():max_coords[:, :, 0].max(),
                                        min_coords[:, :, 2].min():max_coords[:, :, 2].max()]
                            sub_count = f["scan_counts"][min_coords[:, :, 1].min():max_coords[:, :, 1].max(),
                                        min_coords[:, :, 0].min():max_coords[:, :, 0].max(),
                                        min_coords[:, :, 2].min():max_coords[:, :, 2].max()]
                            offset_y = min_coords[:, :, 1].min()
                            offset_x = min_coords[:, :, 0].min()
                            offset_z = min_coords[:, :, 2].min()

                            for sub_y in range(0, group_size):
                                for sub_x in range(0, group_size):
                                    sub_volume[
                                        min_coords[sub_y, sub_x, 1]-offset_y:max_coords[sub_y, sub_x, 1]-offset_y,
                                        min_coords[sub_y, sub_x, 0]-offset_x:max_coords[sub_y, sub_x, 0]-offset_x,
                                        min_coords[sub_y, sub_x, 2]-offset_z:max_coords[sub_y, sub_x, 2]-offset_z] += predictions[y+sub_y, x+sub_x]
                                    sub_count[
                                        min_coords[sub_y, sub_x, 1]-offset_y:max_coords[sub_y, sub_x, 1]-offset_y,
                                        min_coords[sub_y, sub_x, 0]-offset_x:max_coords[sub_y, sub_x, 0]-offset_x,
                                        min_coords[sub_y, sub_x, 2]-offset_z:max_coords[sub_y, sub_x, 2]-offset_z] += 1
                            f["scan_predictions"][min_coords[:, :, 1].min():max_coords[:, :, 1].max(),
                                        min_coords[:, :, 0].min():max_coords[:, :, 0].max(),
                                        min_coords[:, :, 2].min():max_coords[:, :, 2].max()] = sub_volume
                            f["scan_counts"][min_coords[:, :, 1].min():max_coords[:, :, 1].max(),
                                        min_coords[:, :, 0].min():max_coords[:, :, 0].max(),
                                        min_coords[:, :, 2].min():max_coords[:, :, 2].max()] = sub_count
    except:
        print(file)