# mlcourse
生命情報の機械学習入門

この資料は、新学術領域「先進ゲノム支援」中級講習 (2018年11月実施) のために作成しました。

このコースでは、機械学習を利用したクラス分類問題を、実問題を利用して一通り問いてみることに焦点をあてます。
まず前半は、酵母の表現型データベースである、[SCMD (Saccaromices cerevisiae Morphology Database)](http://scmd.gi.k.u-tokyo.ac.jp/datamine/) ([Ohya, Sese et al. PNAS 2005](http://www.pnas.org/content/102/52/19015), [Saito et al. NAR 2004](https://academic.oup.com/nar/article/32/suppl_1/D319/2505341)) のデータを利用して、酵母の画像（および、画像から抽出した特徴量）から、各細胞の細胞周期を同定する機械学習を実施します。後半は、
世界のChIP-seqデータ(転写因子結合サイト実験)を収集し解析したデータベースである[ChIP-Atlas](https://chip-atlas.org/) ([Oki et al. EMBO reports 2018](http://embor.embopress.org/content/early/2018/11/07/embr.201846255)) 
から、一次解析の終了したChIP-seqデータを基に、転写因子結合の有無を予測します。

1. [0章](https://github.com/HumanomeLab/mlcourse/blob/master/0_data_prep_and_visualization.ipynb) : データの準備、可視化
2. [1章](https://github.com/HumanomeLab/mlcourse/blob/master/1_machine_learning_with_features.ipynb) : 特徴量を利用した機械学習(SVM, Random Forest)
3. [2章](https://github.com/HumanomeLab/mlcourse/blob/master/2_deep_learning_for_features.ipynb) : 特徴量を利用した深層学習
4. [3章](https://github.com/HumanomeLab/mlcourse/blob/master/3_deep_learning_for_images.ipynb) : 画像を解析する深層学習
5. [4章](https://github.com/HumanomeLab/mlcourse/blob/master/4_deep_learning_for_sequences.ipynb) : 配列を解析する深層学習
