## Multivariate Seq2Seq LSTM Time Series Prediction
- This project aims to predict the user count in the next 6 hours using the latest 24 hours of data.
- A sequence2sequence LSTM model is built and trained.
- Data can be retrieved from [here.](https://www.kaggle.com/datasets/wolfgangb33r/usercount)

### Requirements & Instructions

- The demonstration of this project can be done on Jupyter notebook or simply run the ```main.py```.
- After running the code, model will be trained and user count for next 24 hours from the latest data point will be predicted (The exact date is 30.12.18 09:00). 
- Besides these you will get the completion time and memory usage of the system will be printed.
- In order to run the ```main.py```, you will need to install ```memory_profiler``` library using ```pip install memory_profiler``` in terminal or if you don't need the system usage and completion time you can exclude the necessary parts from  ```main.py```. 

### Example Output
#### During the training
```
Epoch 1/200
9/9 [==============================] - 27s 375ms/step - loss: 3646.7236 - mae: 47.4411 - root_mean_squared_error: 60.3881 - val_loss: 228606.7969 - 
val_mae: 237.2459 - val_root_mean_squared_error: 478.1284
Epoch 2/200
9/9 [==============================] - 0s 37ms/step - loss: 3795.8381 - mae: 42.5436 - root_mean_squared_error: 61.6104 - val_loss: 2033.0507 - val_mae: 34.4477 - val_root_mean_squared_error: 45.0894
Epoch 3/200
9/9 [==============================] - 0s 32ms/step - loss: 2120.2454 - mae: 33.3576 - root_mean_squared_error: 46.0461 - val_loss: 1623.8368 - val_mae: 29.0819 - val_root_mean_squared_error: 40.2969
Epoch 4/200
9/9 [==============================] - 0s 33ms/step - loss: 1487.9218 - mae: 27.0559 - root_mean_squared_error: 38.5736 - val_loss: 1550.8237 - val_mae: 29.5056 - val_root_mean_squared_error: 39.3805
Epoch 5/200
9/9 [==============================] - 0s 32ms/step - loss: 1495.2534 - mae: 26.7546 - root_mean_squared_error: 38.6685 - val_loss: 754.5371 - val_mae: 17.9088 - val_root_mean_squared_error: 27.4688
Epoch 6/200
9/9 [==============================] - 0s 32ms/step - loss: 955.6856 - mae: 21.6461 - root_mean_squared_error: 30.9142 - val_loss: 10890.6260 - val_mae: 68.2164 - val_root_mean_squared_error: 104.3582
Epoch 7/200
9/9 [==============================] - 0s 33ms/step - loss: 800.8167 - mae: 19.6786 - root_mean_squared_error: 28.2987 - val_loss: 72110.1719 - val_mae: 124.6711 - val_root_mean_squared_error: 268.5334
Epoch 8/200
9/9 [==============================] - 0s 33ms/step - loss: 754.6344 - mae: 19.1311 - root_mean_squared_error: 27.4706 - val_loss: 696.3357 - val_mae: 17.7546 - val_root_mean_squared_error: 26.3882
Epoch 9/200
9/9 [==============================] - 0s 33ms/step - loss: 621.7339 - mae: 17.7975 - root_mean_squared_error: 24.9346 - val_loss: 2834.5398 - val_mae: 20.8972 - val_root_mean_squared_error: 53.2404
Epoch 10/200
9/9 [==============================] - 0s 32ms/step - loss: 631.0298 - mae: 17.7071 - root_mean_squared_error: 25.1203 - val_loss: 1660.8193 - val_mae: 28.2107 - val_root_mean_squared_error: 40.7531
Epoch 11/200
9/9 [==============================] - 0s 33ms/step - loss: 648.6876 - mae: 18.4884 - root_mean_squared_error: 25.4693 - val_loss: 444.6184 - val_mae: 15.0302 - val_root_mean_squared_error: 21.0860
Epoch 12/200
9/9 [==============================] - 0s 34ms/step - loss: 605.5125 - mae: 17.4155 - root_mean_squared_error: 24.6072 - val_loss: 3132839.0000 - val_mae: 770.8163 - val_root_mean_squared_error: 1769.9828
Epoch 13/200
9/9 [==============================] - 0s 36ms/step - loss: 647.0995 - mae: 18.4568 - root_mean_squared_error: 25.4382 - val_loss: 24443684.0000 - val_mae: 3302.4172 - val_root_mean_squared_error: 4944.0557
Epoch 14/200
9/9 [==============================] - 0s 35ms/step - loss: 615.0616 - mae: 17.9509 - root_mean_squared_error: 24.8004 - val_loss: 502.5317 - val_mae: 15.7399 - val_root_mean_squared_error: 22.4172
Epoch 15/200
9/9 [==============================] - 0s 35ms/step - loss: 556.2286 - mae: 17.4933 - root_mean_squared_error: 23.5845 - val_loss: 455.2498 - val_mae: 15.0546 - val_root_mean_squared_error: 21.3366
Epoch 16/200
9/9 [==============================] - 0s 48ms/step - loss: 629.5857 - mae: 18.3508 - root_mean_squared_error: 25.0915 - val_loss: 549.4379 - val_mae: 16.1024 - val_root_mean_squared_error: 23.4401
Epoch 17/200
9/9 [==============================] - 0s 55ms/step - loss: 617.4198 - mae: 17.7978 - root_mean_squared_error: 24.8479 - val_loss: 386.7961 - val_mae: 14.1998 - val_root_mean_squared_error: 19.6671
Epoch 18/200
9/9 [==============================] - 0s 52ms/step - loss: 569.9008 - mae: 17.3064 - root_mean_squared_error: 23.8726 - val_loss: 17004.0488 - val_mae: 92.5679 - val_root_mean_squared_error: 130.3996
Epoch 19/200
9/9 [==============================] - 0s 53ms/step - loss: 515.2878 - mae: 16.4477 - root_mean_squared_error: 22.6999 - val_loss: 518.2984 - val_mae: 16.0357 - val_root_mean_squared_error: 22.7662
Epoch 20/200
9/9 [==============================] - 0s 49ms/step - loss: 575.1736 - mae: 17.8310 - root_mean_squared_error: 23.9828 - val_loss: 478.0816 - val_mae: 15.1853 - val_root_mean_squared_error: 21.8651
Epoch 21/200
9/9 [==============================] - 1s 80ms/step - loss: 585.6851 - mae: 17.5284 - root_mean_squared_error: 24.2009 - val_loss: 522.8951 - val_mae: 15.8659 - val_root_mean_squared_error: 22.8669
Epoch 22/200
9/9 [==============================] - 1s 80ms/step - loss: 591.0372 - mae: 17.9425 - root_mean_squared_error: 24.3113 - val_loss: 353.1317 - val_mae: 13.5151 - val_root_mean_squared_error: 18.7918
Epoch 23/200
9/9 [==============================] - 1s 85ms/step - loss: 600.7141 - mae: 17.7350 - root_mean_squared_error: 24.5095 - val_loss: 5137.3511 - val_mae: 25.5158 - val_root_mean_squared_error: 71.6753
Epoch 24/200
9/9 [==============================] - 1s 57ms/step - loss: 564.5626 - mae: 16.9877 - root_mean_squared_error: 23.7605 - val_loss: 427.3911 - val_mae: 14.6945 - val_root_mean_squared_error: 20.6734
Epoch 25/200
9/9 [==============================] - 0s 50ms/step - loss: 586.4475 - mae: 17.8014 - root_mean_squared_error: 24.2167 - val_loss: 435.2394 - val_mae: 14.7424 - val_root_mean_squared_error: 20.8624
Epoch 26/200
9/9 [==============================] - 1s 84ms/step - loss: 574.1057 - mae: 18.0860 - root_mean_squared_error: 23.9605 - val_loss: 421.1983 - val_mae: 14.6094 - val_root_mean_squared_error: 20.5231
Epoch 27/200
9/9 [==============================] - 0s 48ms/step - loss: 571.3840 - mae: 17.5177 - root_mean_squared_error: 23.9036 - val_loss: 466.6429 - val_mae: 14.8430 - val_root_mean_squared_error: 21.6019
Epoch 28/200
9/9 [==============================] - 1s 63ms/step - loss: 579.3795 - mae: 17.1692 - root_mean_squared_error: 24.0703 - val_loss: 371.9908 - val_mae: 14.0356 - val_root_mean_squared_error: 19.2871
Epoch 29/200
9/9 [==============================] - 0s 53ms/step - loss: 525.8580 - mae: 17.1052 - root_mean_squared_error: 22.9316 - val_loss: 555.2532 - val_mae: 15.4081 - val_root_mean_squared_error: 23.5638
Epoch 30/200
9/9 [==============================] - 0s 48ms/step - loss: 516.8315 - mae: 16.5910 - root_mean_squared_error: 22.7339 - val_loss: 436.0723 - val_mae: 14.6065 - val_root_mean_squared_error: 20.8823
Epoch 31/200
9/9 [==============================] - 0s 42ms/step - loss: 572.5535 - mae: 17.5755 - root_mean_squared_error: 23.9281 - val_loss: 385793.4062 - val_mae: 211.3784 - val_root_mean_squared_error: 621.1227
Epoch 32/200
9/9 [==============================] - 0s 49ms/step - loss: 573.6106 - mae: 17.7857 - root_mean_squared_error: 23.9502 - val_loss: 465.6682 - val_mae: 15.3954 - val_root_mean_squared_error: 21.5793
Epoch 33/200
9/9 [==============================] - 1s 58ms/step - loss: 579.0440 - mae: 17.7973 - root_mean_squared_error: 24.0633 - val_loss: 557893.8125 - val_mae: 283.7376 - val_root_mean_squared_error: 746.9229
Epoch 34/200
9/9 [==============================] - 0s 44ms/step - loss: 584.6718 - mae: 17.8423 - root_mean_squared_error: 24.1800 - val_loss: 11337.2510 - val_mae: 36.3048 - val_root_mean_squared_error: 106.4765
Epoch 35/200
9/9 [==============================] - 0s 42ms/step - loss: 622.7412 - mae: 18.1124 - root_mean_squared_error: 24.9548 - val_loss: 438.4731 - val_mae: 14.8844 - val_root_mean_squared_error: 20.9398
Epoch 36/200
9/9 [==============================] - 0s 38ms/step - loss: 571.8511 - mae: 17.3969 - root_mean_squared_error: 23.9134 - val_loss: 752.6655 - val_mae: 19.1328 - val_root_mean_squared_error: 27.4347
Epoch 37/200
9/9 [==============================] - 0s 39ms/step - loss: 580.5726 - mae: 17.6918 - root_mean_squared_error: 24.0951 - val_loss: 449.5619 - val_mae: 15.0682 - val_root_mean_squared_error: 21.2029
Epoch 38/200
9/9 [==============================] - 0s 39ms/step - loss: 553.2768 - mae: 17.2869 - root_mean_squared_error: 23.5218 - val_loss: 3410211.7500 - val_mae: 302.1314 - val_root_mean_squared_error: 1846.6758
Epoch 39/200
9/9 [==============================] - 0s 39ms/step - loss: 534.9290 - mae: 16.4261 - root_mean_squared_error: 23.1285 - val_loss: 95681432.0000 - val_mae: 1844.2200 - val_root_mean_squared_error: 9781.6885
Epoch 40/200
9/9 [==============================] - 0s 41ms/step - loss: 550.1613 - mae: 17.0236 - root_mean_squared_error: 23.4555 - val_loss: 443.2315 - val_mae: 15.1534 - val_root_mean_squared_error: 21.0531
Epoch 41/200
9/9 [==============================] - 0s 37ms/step - loss: 541.6755 - mae: 17.3183 - root_mean_squared_error: 23.2739 - val_loss: 403.2423 - val_mae: 14.4708 - val_root_mean_squared_error: 20.0809
Epoch 42/200
9/9 [==============================] - 0s 40ms/step - loss: 558.8527 - mae: 17.1659 - root_mean_squared_error: 23.6401 - val_loss: 559.1212 - val_mae: 16.3896 - val_root_mean_squared_error: 23.6457
Epoch 43/200
9/9 [==============================] - 0s 36ms/step - loss: 614.5630 - mae: 17.9941 - root_mean_squared_error: 24.7904 - val_loss: 428.0730 - val_mae: 14.5258 - val_root_mean_squared_error: 20.6899
Epoch 44/200
9/9 [==============================] - 0s 43ms/step - loss: 575.1733 - mae: 17.5502 - root_mean_squared_error: 23.9828 - val_loss: 545556.3125 - val_mae: 165.9621 - val_root_mean_squared_error: 738.6179
Epoch 45/200
9/9 [==============================] - 0s 38ms/step - loss: 573.7281 - mae: 17.5375 - root_mean_squared_error: 23.9526 - val_loss: 407.0167 - val_mae: 14.8164 - val_root_mean_squared_error: 20.1747
Epoch 46/200
9/9 [==============================] - 0s 40ms/step - loss: 580.2368 - mae: 17.7450 - root_mean_squared_error: 24.0881 - val_loss: 430.7133 - val_mae: 14.6982 - val_root_mean_squared_error: 20.7536
Epoch 47/200
9/9 [==============================] - 0s 43ms/step - loss: 590.0773 - mae: 18.0252 - root_mean_squared_error: 24.2915 - val_loss: 427.2059 - val_mae: 14.8033 - val_root_mean_squared_error: 20.6690
Epoch 48/200
9/9 [==============================] - 0s 53ms/step - loss: 568.8588 - mae: 17.5782 - root_mean_squared_error: 23.8508 - val_loss: 424.8350 - val_mae: 14.7498 - val_root_mean_squared_error: 20.6115
Epoch 49/200
9/9 [==============================] - 0s 46ms/step - loss: 565.0748 - mae: 17.5117 - root_mean_squared_error: 23.7713 - val_loss: 434.8040 - val_mae: 14.8049 - val_root_mean_squared_error: 20.8520
Epoch 50/200
9/9 [==============================] - 0s 40ms/step - loss: 569.3813 - mae: 17.3344 - root_mean_squared_error: 23.8617 - val_loss: 474.8227 - val_mae: 15.5122 - val_root_mean_squared_error: 21.7904
Epoch 51/200
9/9 [==============================] - 0s 43ms/step - loss: 578.7734 - mae: 17.5577 - root_mean_squared_error: 24.0577 - val_loss: 424.9032 - val_mae: 15.0299 - val_root_mean_squared_error: 20.6132
Epoch 52/200
9/9 [==============================] - 0s 47ms/step - loss: 533.5460 - mae: 17.0346 - root_mean_squared_error: 23.0986 - val_loss: 438.8103 - val_mae: 14.8164 - val_root_mean_squared_error: 20.9478
Epoch 53/200
9/9 [==============================] - 0s 41ms/step - loss: 545.7215 - mae: 17.2851 - root_mean_squared_error: 23.3607 - val_loss: 409.4117 - val_mae: 14.4663 - val_root_mean_squared_error: 20.2339
Epoch 54/200
9/9 [==============================] - 0s 43ms/step - loss: 562.7502 - mae: 17.4797 - root_mean_squared_error: 23.7224 - val_loss: 472.6561 - val_mae: 15.6409 - val_root_mean_squared_error: 21.7407
Epoch 55/200
9/9 [==============================] - 0s 36ms/step - loss: 577.8367 - mae: 17.4833 - root_mean_squared_error: 24.0382 - val_loss: 421.0396 - val_mae: 14.7000 - val_root_mean_squared_error: 20.5193
Epoch 56/200
9/9 [==============================] - 0s 36ms/step - loss: 563.6177 - mae: 17.3929 - root_mean_squared_error: 23.7406 - val_loss: 402.9697 - val_mae: 14.6363 - val_root_mean_squared_error: 20.0741
Epoch 57/200
9/9 [==============================] - 0s 39ms/step - loss: 557.8784 - mae: 17.5344 - root_mean_squared_error: 23.6194 - val_loss: 398.6303 - val_mae: 14.3165 - val_root_mean_squared_error: 19.9657
Epoch 58/200
9/9 [==============================] - 0s 40ms/step - loss: 591.2000 - mae: 17.8165 - root_mean_squared_error: 24.3146 - val_loss: 552.8566 - val_mae: 17.0167 - val_root_mean_squared_error: 23.5129
Epoch 59/200
9/9 [==============================] - 0s 40ms/step - loss: 551.6798 - mae: 17.0657 - root_mean_squared_error: 23.4879 - val_loss: 29004.2832 - val_mae: 105.0042 - val_root_mean_squared_error: 170.3064
Epoch 60/200
9/9 [==============================] - 0s 37ms/step - loss: 565.2371 - mae: 17.2713 - root_mean_squared_error: 23.7747 - val_loss: 461.5677 - val_mae: 15.7078 - val_root_mean_squared_error: 21.4841
Epoch 61/200
9/9 [==============================] - 0s 35ms/step - loss: 577.3796 - mae: 17.7887 - root_mean_squared_error: 24.0287 - val_loss: 414.2322 - val_mae: 14.6269 - val_root_mean_squared_error: 20.3527
Epoch 62/200
9/9 [==============================] - 0s 39ms/step - loss: 561.2808 - mae: 17.6091 - root_mean_squared_error: 23.6914 - val_loss: 434.2198 - val_mae: 14.8052 - val_root_mean_squared_error: 20.8379
Epoch 63/200
9/9 [==============================] - 0s 40ms/step - loss: 539.9653 - mae: 17.0806 - root_mean_squared_error: 23.2372 - val_loss: 424.2438 - val_mae: 14.6858 - val_root_mean_squared_error: 20.5972
Epoch 64/200
9/9 [==============================] - 0s 34ms/step - loss: 583.0920 - mae: 17.7573 - root_mean_squared_error: 24.1473 - val_loss: 7045.2119 - val_mae: 53.1861 - val_root_mean_squared_error: 83.9358
Epoch 65/200
9/9 [==============================] - 0s 32ms/step - loss: 565.6107 - mae: 17.5808 - root_mean_squared_error: 23.7826 - val_loss: 424.4937 - val_mae: 14.7854 - val_root_mean_squared_error: 20.6032
Epoch 66/200
9/9 [==============================] - 0s 32ms/step - loss: 553.5146 - mae: 17.2227 - root_mean_squared_error: 23.5269 - val_loss: 427.3980 - val_mae: 15.0922 - val_root_mean_squared_error: 20.6736
Epoch 67/200
9/9 [==============================] - 0s 33ms/step - loss: 592.2845 - mae: 18.1739 - root_mean_squared_error: 24.3369 - val_loss: 412.5522 - val_mae: 14.6598 - val_root_mean_squared_error: 20.3114
Epoch 68/200
9/9 [==============================] - 0s 32ms/step - loss: 562.1278 - mae: 17.6700 - root_mean_squared_error: 23.7092 - val_loss: 405.0460 - val_mae: 14.5498 - val_root_mean_squared_error: 20.1258
Epoch 69/200
9/9 [==============================] - 0s 36ms/step - loss: 620.4318 - mae: 17.8460 - root_mean_squared_error: 24.9085 - val_loss: 426.4862 - val_mae: 14.8356 - val_root_mean_squared_error: 20.6515
Epoch 70/200
9/9 [==============================] - 0s 35ms/step - loss: 579.0109 - mae: 17.9412 - root_mean_squared_error: 24.0626 - val_loss: 420.2821 - val_mae: 14.8151 - val_root_mean_squared_error: 20.5008
Epoch 71/200
9/9 [==============================] - 0s 38ms/step - loss: 565.4163 - mae: 17.7073 - root_mean_squared_error: 23.7785 - val_loss: 429.4040 - val_mae: 14.7643 - val_root_mean_squared_error: 20.7221
Epoch 72/200
9/9 [==============================] - 0s 42ms/step - loss: 553.0167 - mae: 16.9727 - root_mean_squared_error: 23.5163 - val_loss: 385.5953 - val_mae: 14.2399 - val_root_mean_squared_error: 19.6366
```
#### After the training
```
Training completed ...
30.12.18 09:00:00 User Count: 69
```
#### When running is done
```

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
     6    269.4 MiB    269.4 MiB           1   @profile
     7                                         def main():
     8    270.3 MiB      0.8 MiB           1       data = preprocess.get_data()
     9    270.4 MiB      0.1 MiB           1       train_input_seq, test_input_seq, train_output_seq, test_output_seq = preprocess.create_train_test_data()
    10    286.2 MiB     15.8 MiB           1       seq2seq = model.Seq2SeqModel(train_input_seq,train_output_seq)
    11    448.6 MiB    162.4 MiB           1       seq2seq.fit_model()
    12    448.6 MiB      0.0 MiB           1       print("Training completed ...")
    13    453.2 MiB      4.5 MiB           1       print("30.12.18 09:00:00 User Count: {}".format(int(seq2seq.predict_next_24hours(data[145:]))))
    14    453.2 MiB      0.0 MiB           1       pass


System complete time: 63.53 seconds
```
