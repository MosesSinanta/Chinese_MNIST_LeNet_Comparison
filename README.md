
# Chinese MNIST LeNet Comparison

This project evaluates and compares the performance of LeNet-4 and LeNet-5 Convolutional Neural Network architectures on the Chinese MNIST dataset for handwritten digit classification.

#### Disclaimer
This project is not the first of its kind, and there are many similar projects on the same topic. It is created solely for educational purposes.



## Deployment

#### 1. Change the model type you want to run by commenting line 22 / 23 in main.py
```
#model = lenet4_model(input_shape = (32, 32, 1), num_classes = num_classes)
model = lenet5_model(input_shape = (32, 32, 1), num_classes = num_classes)
```

#### 2. Change the JSON file output name in line 54 in main.py (optional)
```
with open("LeNet-5 results.json", "w") as file:
```

#### 3. Run main.py in terminal
```
python main.py
```



## Results

Past-run results can be seen in ```LeNet-4 results.json``` and ```LeNet-5 results.json```

This project also includes documentation in PDF.



## Authors

#### [MosesSinanta](https://github.com/MosesSinanta/)
A random nerd who likes doing fun projects.
