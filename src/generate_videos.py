"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Generates multiple videos given a model and saves them as video files using ffmpeg

Usage:
    generate_videos.py [options] <model> <output_folder>

Options:
    -n, --num_videos=<count>                number of videos to generate [default: 10]
    -o, --output_format=<ext>               save videos as [default: gif]
    -f, --number_of_frames=<count>          generate videos with that many frames [default: 16]

    --ffmpeg=<str>                          ffmpeg executable (on windows should be ffmpeg.exe). Make sure
                                            the executable is in your PATH [default: ffmpeg]
"""
import cv2
import os
import docopt
import torch

from trainers import videos_to_numpy

import subprocess as sp

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def save_video(video, filename):
    cap = cv2.VideoCapture(video)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(True):

        ret, frame = cap.read()
  
        if ret == True:
            out.write(frame)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break 
      

if __name__ == "__main__":
    args = docopt.docopt(__doc__)

    generator = torch.load(args["<model>"], map_location={'cuda:0': 'cpu'})
    generator = recursion_change_bn(generator)
    generator.eval()
    num_videos = int(args['--num_videos'])
    output_folder = args['<output_folder>']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_videos):
        v, _ = generator.sample_videos(1, int(args['--number_of_frames']))
        video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        #save_video(args["--ffmpeg"], video, os.path.join(output_folder, "{}.{}".format(i, args['--output_format'])))
        cap = cv2.VideoCapture(video)

        if (cap.isOpened() == False):
            print("Unable to read camera feed")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

        while(True):

            ret, frame = cap.read()
  
            if ret == True:
                out.write(frame)
                cv2.imshow('frame',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break 