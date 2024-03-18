```python
import cv2
import os
import numpy as np

data_train = np.empty((100000,3,32,32),dtype="uint8")
train_imgs = os.listdir('./(原圖)IKEA_shark')
for i in range(0,len(train_imgs)):
    img = cv2.imread('./(原圖)IKEA_shark/'+train_imgs[i])
    if img is None:#如果圖片數據是空的，就跳過
        continue
    new_img = cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR)
    data_train[i,:,:,:] = [new_img[:,:,0],new_img[:,:,1],new_img[:,:,2]]
    cv2.imwrite('./IKEA_shark_train/new_'+train_imgs[i], new_img)
```

主要重點：

```python
# 利用cv2.resize()，將原圖片強制轉成32x32的大小，加速讀取資料的時間
new_img = cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR)
```



