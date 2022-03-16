Se tienen 3 ramas donde:
> Hay una rama serial que es el código base sin paralelizar-
> La rama de prueba paralelización 2 se optimiza con openCL la porción de código del computo de las neuronas excitatorias
> La rama de prueba paralelización 3, se intenta paralelizar el computo de la región excitatoria y de las neuronas de entrada.

### Se realiza un experimento comparando 100 muestras en donde se obtiene un tiempo de 157 segundos para el código paralelo y 104 segundos para el código serial. Se verifica constantemente que el algoritmo paralelo que se está modificando coincida con la versión serial para verificar errores de códificación.
