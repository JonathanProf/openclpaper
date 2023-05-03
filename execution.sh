NEURONS=1600

rm "classification/labelsVIM3openCL"$NEURONS"N_64ms.csv"

clear
g++ -Wall main.cpp functions.cpp -o  main.out -lOpenCL -std=c++11

DATE=$(eval date --iso-8601='minutes')
fileOut='results/log_'$NEURONS'N_64ms_'$DATE'.txt'

#<<comment
#for i in 2 4 5 8 10 16 20 40 50 80 100;
for i in 8;
do
    echo "start test with "$NEURONS"N and $i WG"
    echo "start test with "$NEURONS"N and $i WG" >> $fileOut

    date
    date >> $fileOut

    time ./main.out $i >> $fileOut

    echo ""
    echo "" >> $fileOut

    echo "=====     =====     =====     ====="
    echo "=====     =====     =====     =====" >> $fileOut

    echo "Comparison Workstation OpenCL vs. openCL VIM3:"
    echo "Comparison Workstation OpenCL vs. openCL VIM3:" >> $fileOut

    python3 PythonCode/file_comparison.py
    python3 PythonCode/file_comparison.py >> $fileOut

    echo "=====     =====     =====     ====="
    echo "=====     =====     =====     =====" >> $fileOut

    date
    date >> $fileOut

    echo ""
    echo "" >> $fileOut

    rm "classification/labelsVIM3openCL"$NEURONS"N_64ms.csv"

    echo "Net with $i WG: test done"
    echo "Net with $i WG: test done" >> $fileOut

    echo ""
    echo "" >> $fileOut
    
    echo "-----"
    echo "-----">> $fileOut
done
rm main.out
#comment