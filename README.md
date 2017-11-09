# 講義第４回

## 目的

+ 実際に学習したモデルを使ったシステムを作ることを通して、DeepLearningを使ったシステムの作り方について学ぶ

## 概要

+ DCGANの復習とpix2pixの復習(10分)
+ デモを動かす(30分)
+ 質問など

## DCGANとpix2pixの復習

+ GAN
  + Generator(生成器)とDiscriminator(判別器)を競い合わせて、お互いの精度を高め合わせる手法

+ DCGAN
  + ノイズから画像を生成する
  + Genrator: ノイズから画像を生成する
  + Discriminator: 画像がGeneratorから生成された偽物か、本物の画像かを判別する

+ pix2pix
  + 画像から別の画像を生成する
  + Genrator: 画像から別の画像を生成する
  + Discriminator: 画像がGeneratorから生成された偽物か、本物の画像かを判別する

## DCGANのデモを動かしてみる

1. Github`https://github.com/hiko2msp/gan_view`にアクセスし、forkボタンを押す
1. Herokuにアクセスする
1. Herokuのアプリケーションを作成する
1. Deploy -> Githubの連携 -> forkしたGithubレポジトリを選択 -> masterブランチをManual Deploy -> 完了を待つ
1. Open appボタンを押して起動

## pix2pixのデモを動かしてみる

1. Github`https://github.com/hiko2msp/gan_view`にアクセスし、forkボタンを押す
1. Herokuにアクセスする
1. Herokuのアプリケーションを作成する
1. Deploy -> Githubの連携 -> forkしたGithubレポジトリを選択 -> pix2pix\_tensorflowブランチをManual Deploy -> 完了を待つ
1. Open appボタンを押して起動

## デモの仕組みを説明

+ Heroku
+ Github
+ Flask




