import Augmentor
import os
import pp
from multiprocessing import Pool

def augmentor(dir, out_dir):
    p = Augmentor.Pipeline(dir, output_directory=out_dir)
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=10)
    p.zoom(probability=0.2, min_factor=1.0, max_factor=1.2)
    p.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.5)
    #p.flip_top_bottom(probability=0.5)#
    #p.crop_random(probability=0.5, percentage_area=0.9)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.skew(probability=0.5, magnitude=1)
    p.resize(probability=1, width=224, height=224)
    p.sample(500)

def precessing(root, new_root):
    all_dir = []
    all_new_dir = []
    arg = []
    for r, subdir, files in os.walk(root):
        if 0 == len(files):
            continue
        new_sub_dir = r.split('/')[-1]
        new_dir = new_root + '/' + new_sub_dir
        if(False == os.path.exists(new_dir)):
            os.mkdir(new_dir)
        augmentor(r, new_dir)


    # ppservers = ()
    # job_server = pp.Server(ppservers=ppservers)
    # jobs = [(input, job_server.submit(augmentor, (all_dir[i], all_new_dir[i], ))) for i in xrange(len(all_dir))]


if __name__ == "__main__":
    precessing('/home/zyh/PycharmProjects/baidu_dog/crop_train', '/home/zyh/PycharmProjects/baidu_dog/new_crop_train_augment')

