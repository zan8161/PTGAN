github: 
- https://github.com/zan8161/PTGAN.git
	- demo code is writen in webui branch，main branch is used to train the reid model.

```bash
git clone -b webui https://github.com/zan8161/PTGAN.git
cd PTGAN
```

# 執行說明

## Environment
Python version : __3.10.12__
1. ```pip install -r demo_requirements.txt```
2. install torch,  torchvision ( __cuda support__ ) 
    - https://pytorch.org/get-started/locally/
    - e.g. ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```


Download reid model(The size of the .pth file is greater than 300MB) :  https://drive.google.com/file/d/1kAeqozPfy7ODdvkr04X0hnvunMsCsd_P/view?usp=sharing
Then put the .pth file to ```./checkpoints/stage2/transreid_256/v1/```


## Our ReID Pipe:
<img src="./docs/reid_demo.jpg">

## Demo 

Demo說明:
Gallery 所有車輛的 features 都預先用 ```preprocess.py``` 算好了，存放在 : ```./precaculate/gallery```

執行:
```bash 
python ui.py
```
會在 terminal 產生local host & public host(這個最多跑72 hrs):
e.g.
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://a1d67dd74f967e07aa.gradio.live
```
使用瀏覽器開啟任一 URL ，就會跳出web UI介面

web UI 介面使用說明:
1. 左邊有一個可以選擇 query 車輛圖片之介面，按下他，並從
```./VehicleData/query/``` 下方挑一輛想要 ReID 的車輛
2. 設置 topk 之值，最大展示的 topk = 10, 建議設 5~6 
3. 選擇要用哪種計算特徵距離的方式 (Distance type):
    - cosine distance
    - re_ranking: 
        - CVPR 2017 一篇論文之 Reranking 方法 
        - (Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.)
        - __使用這個要跑大概 20幾秒，前面的(cos distance)只要 1 秒左右__
4. 以上都設置好了之後，按下 Match，就會開始計算距離，算完後會在下面顯示共8個 camera 中與 query 車輛 topk $(\leq 10)$ 特徵相似之車輛

可以重複選好幾次車，只要確認 __query__, __topk__, __Distance_type__ 都有設置，按下 Match 就能夠針對這3個設置給出結果。

結束就關掉瀏覽器頁面，並直接 __回 terminal 按下 Ctrl+C__ 或是 __直接關掉該 terminal__ 

### Demo 影片 :
https://drive.google.com/file/d/1r1ddUSgRAxANteRpAwl5FvD7XXkya8w1/view?usp=drive_link

## ReID model 建立 與 使用 :
在 ./models/ViTReID.py 裡面，有一個 ```TransReIDBase_Inference```
的 class，就是 ReID model 

config 檔 :  ./config/transreid_256_veri_gan.yml

reid 一台車的 demo : 
python model_using_example.py 

