#!/bin/bash
## for i in $(ls conf/ir/next/ -1 | sed -e 's/\.yml$//')
## do
##   #echo "------ conf/ir/next/$i.yml"
##   screen -S $i -dm ./run_method.sh conf/ir/next/$i.yml
## done
##i=lastfm_rules
##bash -c 'conda activate py37 && ./run_method.sh conf/ir/next/$i.yml'



screen -S 30music_knn2 -dm bash -c './run_method.sh conf/ir/next/30music_knn2.yml'
screen -S 30music_knn -dm bash -c './run_method.sh conf/ir/next/30music_knn.yml'
screen -S 30music_rules -dm bash -c './run_method.sh conf/ir/next/30music_rules.yml'
screen -S lastfm_knn2 -dm bash -c './run_method.sh conf/ir/next/lastfm_knn2.yml'
screen -S lastfm_knn -dm bash -c './run_method.sh conf/ir/next/lastfm_knn.yml'
screen -S lastfm_rules -dm bash -c './run_method.sh conf/ir/next/lastfm_rules.yml'
screen -S nowplaying_knn2 -dm bash -c './run_method.sh conf/ir/next/nowplaying_knn2.yml'
screen -S nowplaying_knn -dm bash -c './run_method.sh conf/ir/next/nowplaying_knn.yml'
screen -S nowplaying_rules -dm bash -c './run_method.sh conf/ir/next/nowplaying_rules.yml'

screen -S 30music_neural -dm bash -c './run_method.sh conf/ir/next/30music_neural.yml'
screen -S lastfm_neural -dm bash -c './run_method.sh conf/ir/next/lastfm_neural.yml'
screen -S nowplaying_neural -dm bash -c './run_method.sh conf/ir/next/nowplaying_neural.yml'
