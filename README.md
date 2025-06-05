#### MIT YOLO 

- TIFF画像には対応していないため, 画像フォーマットを[".jpeg", ".jpg", ".png"]のいずれかに変更

- 一般的なYOLOフォーマットと異なり, MIT YOLOはbbox=[left, top, right, bottom]
- 座標はすべてwidth, heightに対する相対値[0, 1]

- logging_utils.py: 110を修正  
    https://github.com/MultimediaTechLab/YOLO/issues/185
    ```python
    修正前: metrics.pop("v_num")
    修正後: metrics.pop("v_num", None)
    ```

- DataLoaderにshuffleの引数を渡すよう変更  
    https://github.com/MultimediaTechLab/YOLO/issues/194
    ```python
    shuffle=data_cfg.shuffle
    ```

- model/yolo.py内のcreate_model()において, weight_pathにckptファイルを指定できるよう修正
    https://github.com/MultimediaTechLab/YOLO/issues/136

- inference時のDataloaderを自作(StreamDataLoaderを用いない)
    StreamDataLoaderを使うと, フォルダ内の画像の一部しか推論されない

- inference時にTrainerを用いない方法を使う
    lazy.pyのtrainer.predict()を使う方法では正しい結果が出力されない

- YOLOv9-mのモデル構造を修正(v9-m.yaml 58行目)
    https://github.com/MultimediaTechLab/YOLO/issues
    ```
    - AConv:
        args: {out_channels: 180}
    ```
- 学習率を下げる
    0.01 -> 0.0001