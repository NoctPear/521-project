## CNN Model
### CNN Base
- Epoch 20/20
80/80 ━━━━━━━━━━━━━━━━━━━━ 25s 315ms/step - accuracy: 0.9702 - loss: 0.0753 - val_accuracy: 0.9365 - val_loss: 0.1751
40/40 ━━━━━━━━━━━━━━━━━━━━ 4s 108ms/step - accuracy: 0.8925 - loss: 0.4098

- input size：224 × 224 × 3
- optimizer：Adam
- learning raate: 0.001
- Loss function：binary_crossentropy
- batch size：32
- epochs：20
- Dropout：0.5

✅ CNN Base Test Accuracy: 0.9016
![CNN_Base](https://github.com/user-attachments/assets/ee1bc653-33d7-4bc8-816d-bed08358300c)


### CNN fine-tune

- batch size：32
- epochs : 30
- dropout_rate : 0.7

- Adam-0.001: Test Accuracy = 0.9164
![Adam-0 001_performance](https://github.com/user-attachments/assets/6c261c59-fdef-4156-8f30-0f201fa789d3)

- Adam-0.0005: Test Accuracy = 0.9109
![Adam-0 0005_performance](https://github.com/user-attachments/assets/85fe7d3b-2564-4324-b4cb-dc93ffc6e6a5)

- RMSprop-0.001: Test Accuracy = 0.9008
![RMSprop-0 001_performance](https://github.com/user-attachments/assets/06fdacf1-d1f5-4aed-aa74-0a1387069a3a)

- SGD-0.01: Test Accuracy = 0.9203
![SGD-0 01_performance](https://github.com/user-attachments/assets/10f7c88c-0b67-45c4-ad58-9d9b401de716)

- optimizer_accuracy_comparison

![optimizer_accuracy_comparison](https://github.com/user-attachments/assets/dde241f0-bb3c-4d12-ad62-e503a4466183)


