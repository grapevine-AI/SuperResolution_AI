# SuperResolution_AI

私の開発した超解像AIの訓練コードです。[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)をベースに開発しています。  
データセットにDIV2Kを使用していますが、ライセンスの関係上アップロードはしません（このようなファイルが他にもいくつか存在します）。  
学習済みモデルによる[デモ](https://colab.research.google.com/drive/1SVp1TQERLCVB5XMAEL7eRt48bMuYe9_m?usp=sharing)も用意していますので、コードだけではなく性能に興味のある方はご利用ください。  
なおiPhoneのカメラで撮影した写真と同等の解像度ならばエラーにならないように調整済みですが、colabの制約上、4K解像度を超える画像は本来の性能で推論ができません。  
また、デモ用ネットワークと訓練コードのネットワークは一部のアーキテクチャが異なります。  
  
## 各プログラムファイルの説明  
base_net.py: ベースモデルを定義しています。  
dockerfile: docker desktopを使用して学習を行うためのものです。  
inference.py: 推論部のコードです。  
my_loader.py: マルチスレッドでデータ生成を行います。  
pretraining.py: L1ロスのみでの事前学習を行います。GPU訓練部は別スレッドに隔離し、データ生成と訓練をパイプライン処理することで、100パーセントに近い稼働率を実現します。  
SR_util.py: データ生成に用いる関数（たとえば、論文中のgeneralized gauss distributionなど）の定義部です。  
Torch_to_TRT.py: PyTorchモデルを経由し、モデルを[TensorRT](https://github.com/NVIDIA-AI-IOT/torch2trt)化します。  
training.py: GANを用いた訓練部です。こちらも複数スレッドパイプライン処理に対応しています。  
vanilla_loader.py: 私のカスタマイズが加わっていない、Real-ESRGAN本家をできる限り再現したデータジェネレータです。マルチスレッド対応。  
videoSR_TRT.py: TensorRTで高速に動画を超解像します。目安としてはINT8で5倍、FP16で2.5倍高速化します。ただし、変換の時点でかなりのVRAMが必要です。  
videoSR.py: 動画に超解像を適用します。元動画のデコード、超解像、エンコードの3スレッドでパイプライン処理するので、エンコード・デコードの際にGPUが停止することはありません。エンコードにシスコ社のopenH264、音声データの移植にffmpegを使用します。  
