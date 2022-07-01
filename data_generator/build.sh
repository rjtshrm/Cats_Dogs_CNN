#!/bin/bash

build_name=$1

data_generator/"$build_name"


EXPOSE 5000

docker build -t data_generator:"$build_name" .