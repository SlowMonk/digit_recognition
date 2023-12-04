# Train the calssificaiton 

```
python3 train_classification.py --train_path {train_data_path} --test_path {test_data_path} --extra_path {extra_data_path} --device {device path} --epoch {num_epoch}
```

you must give train_path and extra_path for the training 


# Train the outlier(VAE)

```
python3 train_outlier.py --train_path {train_data_path}  --test_path {test_data_path} --extra_path {extra_data_path} --device {device path} --epoch {num_epoch} --gray False
```

you must give train_path, extra_path for the right training


# Inference 
There is 5 images in inference called test1.png, test2.png, test3.png, test4.png, test5.png. 

```
python3 run.py --vgg_path {vgg_weight_path} --vae_path {vae_weight_path} 
```

I included the 
    vgg weight :https://gatech.box.com/s/tswhc58f47buqm4ueuh5tft66mulhya8
    vae_weight: https://gatech.box.com/s/00krrha4y635zkz4ppkgclkax60hicwf

You must download and include in the path above