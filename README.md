# 目的

pix2pixで学習したモデルを確認するためのWebサービスを構築します

# 準備

## Webサービスの構築

+ 前提
  + Herokuのアカウントがある
  + Githubのアカウントがある

+ Githubでこのレポジトリをforkする
+ HerokuのApplicationを作成する
+ HerokuのGithub連携でforkしたレポジトリを選択
  + branchによって機能が異なります
  + pix2pix\_tensorflowはTensorflow用のpix2pixのモデルを動かすためのWebサービスを作ることができます

## モデルの学習

+ 第11回修正コード.zipを展開する
  + 5studyai\_11m\_03.py
  + dropout\_demo01.py 
  + facade\_dataset2.py
  + の三つのファイルが展開されることを確認する
+ 第二期第11回予習用資料.zipを展開する
  + 第二期第11回予習用資料フォルダに含まれるfacade.zipを展開する
    + 展開して出てきたfacedeのフォルダを第11回修正コードのフォルダに移動する
+ 5studyai\_11m\_03.pyのファイルの修正
  + 160行目あたりにsaverオブジェクトを生成するプログラムを追加

    ```
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ```

  + 195行目あたりにモデルの保存を行うプログラムを追加

    ```
    if epoch % 30 == 0:
        # save model
        saver.save(sess, './out_models_pix2pix/model.ckpt', global_step=epoch)
        #draw image
        pylab.rcParams['figure.figsize'] = (28.0, 28.0)
    ```

+ 学習してモデルを生成する

  ```
  $ cd 第11回修正コード
  $ python 5studyai_11m_03.py
  ```
