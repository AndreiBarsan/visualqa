#!/usr/bin/env bash

# install unzip to run this

function unzipRemove() {
   find . -name '*.zip' -exec sh -c 'unzip -d `dirname {}` {}' ';'
   rm *.zip
}

mkdir data
cd data
mkdir vqa
cd vqa

# download Questions

mkdir Questions
cd Questions
wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.ziphttp://visualqa.org/data/abstract_v002/vqa/Questions_Val_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Questions_Test_abstract_v002.zip
unzipRemove
cd ..

# download Annotations

mkdir Annotations
cd Annotations
wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip
wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Train_abstract_v002.zip
wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Val_abstract_v002.zip
unzipRemove
cd ..

# download Images

mkdir Images
cd Images
mkdir mscoco
cd mscoco

mkdir train2014
cd train2014
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzipRemove
cd ..

mkdir val2014
cd val2014
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzipRemove
cd ..

mkdir test2015
cd test2015
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzipRemove
cd ..

cd ..
mkdir abstract_v002
cd abstract_v002

mkdir train2015
cd train2014
wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip
unzipRemove
cd ..

mkdir val2015
cd val2014
wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip
unzipRemove
cd ..

mkdir test2015
cd test2015
wget http://visualqa.org/data/abstract_v002/scene_img/scene_img_abstract_v002_test2015.zip
unzipRemove
cd ..

cd..
cd ..
mkdir Preprocessed
