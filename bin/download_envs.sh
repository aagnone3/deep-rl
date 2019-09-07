#!/usr/bin/env bash
set -eou pipefail

urls=(
    https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
    https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
    https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
)
folders=(
    p1_navigation
    p2_continuous-control
    p3_collab-compet
)
zips=(
    Banana_Linux_NoVis.zip
    Reacher_Linux_NoVis.zip
    Tennis_Linux_NoVis.zip
)

for i in $(seq 0 2)
do
    pushd ${folders[$i]}
    wget ${urls[$i]}
    unzip ${zips[$i]}
    popd
done
