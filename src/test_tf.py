import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import time


from architechtures.generative_inpainting.inpaint_model import InpaintCAModel



parser = argparse.ArgumentParser()

parser.add_argument('--model', default='generative_inpainting', type=str,
                    help='The model we use')
parser.add_argument('--dataset', default='', type=str,
                    help='The folder with masks and images')
parser.add_argument('--task', default='', type=str,
                    help='The task we are testing')



if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()


    input_images = np.array([])
    i = 0

    for file in os.listdir(f"../data/{args.dataset}/images"):
        id = file.split('.')[0]
        image = cv2.imread(f"../data/{args.dataset}/images/{file}")
        mask = cv2.imread(f"../data/{args.dataset}/masks/{id}.png")
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        print('Shape of input image: {}'.format(input_image.shape))

        if(i == 0):
            input_images = input_image
            i = 1
        else:
            print(f"We concatenate shapes : {input_image.shape} and {input_images.shape}")
            input_images = np.concatenate([input_images, input_image], axis=0)

    print(f"Input images shape : {input_images.shape}")
    t1 = time.time()
    model = InpaintCAModel()


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_images, dtype=tf.float32)
        print(f"Input shape : {input_image.shape}")
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(f"../experiments/Task{args.task}/models", from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        t2 = time.time()
        print(f'Model loaded in {t2-t1}s')
        result = sess.run(output)
        t3 = time.time()
        print(f'Predictions done in {t3-t2}s')
        print(f"result shape : {result.shape}")
        for id,im in enumerate(result):
            cv2.imwrite(f"../experiments/Task{args.task}/results/{id}.jpg", im[:, :, ::-1])

    print(f"Total time : {t3-t1}")
