#!/bin/bash

build_name=$1

docker build -t data_generator:"$build_name" .