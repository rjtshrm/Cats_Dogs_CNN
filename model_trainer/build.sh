#!/bin/bash

build_name=$1

docker build -t trainer:"$build_name" .