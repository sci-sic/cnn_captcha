# cnn_captcha
Brief pipeline to train a cnn to break alphanumeric captcha using Python and keras to re-train Google's InceptionV3.

# Creating data set

`python gen_cap.py -du --npi=5 -n 1`

# Train network

`python keras_inception.py`
