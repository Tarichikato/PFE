# PFE

You can test with the model you want

1 : First download the models from [HERE](https://drive.google.com/drive/folders/14NAl7_aFo5CKsfZtikx-7ugfuKobisI8?usp=sharing)  and put them in TaskXXX/models

2 : set up the database you want to test like this :


    database
       images
            001.jpg
            002.jpg
            
        masks
            001.png
            002.png
            
You can give your files the names you want but corrsponding image and mask need to have matching names. For now the files extensions are not negotiable you need to have .jpg    for the images and .png for masks but if may evolve in the futur.
Be carefull your masks need to be at 255 where your cloud is and 0 where it isn't. It may also change in the futur.
     
 3 : move in src folder and execute : python test_tf.py --dataset database --task XXX
 
 Your reconstructed images are in ../experiments/TaskXXX/results


To test with the demo images execute : python test_tf.py --dataset test1 --task 001

I will soon add a link with a database to download
