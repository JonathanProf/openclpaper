#!/bin/bash

clear
rm main.out ./classification/labelsVIM3openCL400N_64ms.csv
g++ -Wall main.cpp functions.cpp -o  main.out -lOpenCL -std=c++11


file_out=log_400N_64ms_test_23_2_binInput.txt
for i in 5
do
		echo "******************************"
        echo "* Start probe with $i groups *"
        echo "******************************"
        date
        echo "******************************" >> $file_out
        echo "* Start probe with $i groups *" >> $file_out
        echo "******************************" >> $file_out
        date >> $file_out
        time ./main.out $i >> $file_out
        echo "" >> $file_out
        echo "=====     =====     =====     =====" >> $file_out
        echo "Comparison betweenSerial vs. openCL codes 400N:" >> $file_out
        python3 file_comparison.py >> $file_out
        echo "=====     =====     =====     =====" >> $file_out
        date >> $file_out
        echo "****************************" >> $file_out
        echo "* End probe with $i groups *" >> $file_out
        echo "****************************" >> $file_out
        echo "-" >> $file_out
        echo "-" >> $file_out

        echo ""
        echo "=====     =====     =====     ====="
        echo "Comparison betweenSerial vs. openCL codes 400N:"
        python3 file_comparison.py
        echo "=====     =====     =====     ====="
        date
        echo "****************************"
        echo "* End probe with $i groups *"
        echo "****************************"
        
		rm ./classification/labelsVIM3openCL400N_64ms.csv
		echo "-"
        echo "-"
done

rm main.out
