## CNN Model
### CNN Base
- Epoch 20/20
80/80 ━━━━━━━━━━━━━━━━━━━━ 25s 315ms/step - accuracy: 0.9702 - loss: 0.0753 - val_accuracy: 0.9365 - val_loss: 0.1751
40/40 ━━━━━━━━━━━━━━━━━━━━ 4s 108ms/step - accuracy: 0.8925 - loss: 0.4098

✅ CNN Base Test Accuracy: 0.9016
![CNN_Base](https://github.com/user-attachments/assets/ee1bc653-33d7-4bc8-816d-bed08358300c)


### CNN fine-tune


- Best test accuracy: 0.9250 with config: optimizer=Adam, epochs=30, batch_size=32, dropout=0.7
- 40/40 ━━━━━━━━━━━━━━━━━━━━ 7s 177ms/step - accuracy: 0.9181 - loss: 0.3931 

- ✅ Best Model Test Accuracy: 0.9250

  
![CNN_best-1](https://github.com/user-attachments/assets/c886ed4e-7ffe-49cd-b0f3-0a1bc384097a)

![best_optimizer-1](https://github.com/user-attachments/assets/8b23b2f6-2abf-4884-a27e-d60dd64ea60f)

