__kernel void mainKernel(			int numNeur,												// constant
									int timeS,													// constant
									int indexWin,												// variable
							   		__global float *theta,									// read only
							   		__global float *weightsXeAe,						// read only
								   	__global float *weightsAeAi, 						// read only
							   		__global float *weightsAiAe, 						// read only
									__global unsigned int *input_sample,						// read only	
							   		__global bool *spikesXePre,						// read-write
							   		__global bool *spikesXePos,						// read-write
							   		__global bool *spikes_Ae_Ai_pre, 				// read-write
								   	__global bool *spikes_Ae_Ai_pos, 				// read-write
								   	__global bool *spikes_Ai_Ae_pre, 				// read-write
							   		__global bool *spikes_Ai_Ae_pos, 				// read-write
							   		__global unsigned short int *spike_count, 	// read-write
							   		__global unsigned short int *assignments,	// read only
							   		__global unsigned short int *digits,				// read-write
							   		__global float *vE, 										// read-write
								   	__global float *vI, 										// read-write
							   		__global int *refrac_countE,						// read-write
								   	__global int *refrac_countI)							// read-write
{

	int threadNum = get_global_id(0);

	unsigned int datoAnalisis = 0;
    unsigned int resultOP = 0;
    unsigned int desplazamiento = 0;
	
	datoAnalisis = ( timeS < 32 ) ? input_sample[ threadNum*2 ] : input_sample[ threadNum*2+1 ];
    desplazamiento = ( timeS < 32 ) ? (31 - timeS) : (63 - timeS);
    resultOP = datoAnalisis & (1 << desplazamiento);
    resultOP >>= desplazamiento;

    // Se verifica que la información sea binaria
    //assert( resultOP == 0 or resultOP == 1);
    spikesXePre[threadNum] = resultOP;

	/*
	int indexSample = threadNum+timeS*784;
	if ( input_sample[indexSample] == true ) {
		spikesXePre[threadNum] = 1;
	}
	else{
		spikesXePre[threadNum] = 0;
	}
	*/
	
	if( numNeur+threadNum < 784 ){
	/*	
		if ( input_sample[ indexSample + numNeur] == 1 ) {
			spikesXePre[numNeur+threadNum] = 1;
		}
		else{
			spikesXePre[numNeur+threadNum] = 0;
		}
	*/

		datoAnalisis = ( timeS < 32 ) ? input_sample[ (threadNum+numNeur)*2 ] : input_sample[ (threadNum+numNeur)*2+1 ];
		desplazamiento = ( timeS < 32 ) ? (31 - timeS) : (63 - timeS);
		resultOP = datoAnalisis & (1 << desplazamiento);
		resultOP >>= desplazamiento;

		// Se verifica que la información sea binaria
		//assert( resultOP == 0 or resultOP == 1);
		spikesXePre[threadNum+numNeur] = resultOP;
	}
	
	// 1. Update E neurons
	float vSyn = 0.0f;
	vSyn = 0.0f;
	for (int i = 0; i < numNeur; ++i) {
		vSyn += spikes_Ai_Ae_pos[ i ] ? weightsAiAe[ threadNum * numNeur + i ] : 0.0f ;
	}
	
	//barrier( CLK_GLOBAL_MEM_FENCE );
	
	for (int i = 0; i < 784; ++i) {
		vSyn += spikesXePos[ i ] ? weightsXeAe[ i * numNeur + threadNum ] : 0.0f ;
	}
	
	//barrier( CLK_LOCAL_MEM_FENCE );
	
	
	vE[threadNum] = 0.9900498390197753906251 * (vE[threadNum] - (-65.0)) + (-65.0);
	
	
	
	if( refrac_countE[threadNum] <= 0 )
		vE[threadNum] += vSyn;


	refrac_countE[threadNum] -= 1;

	
	spikes_Ae_Ai_pre[threadNum] = (vE[threadNum] > ( -52.0 + theta[threadNum])) ? 1 : 0;
	
	//resetPotentialEx(spikesE);
	if( spikes_Ae_Ai_pre[threadNum] == 1 ){
		refrac_countE[threadNum] = 5;
		vE[threadNum] = -60.0;
	}
	
// =================
// REVISAR MODIFICACIÓN
// =================
	if( spikes_Ae_Ai_pre[threadNum] != 0){
		indexWin = ( threadNum < indexWin ) ? threadNum : indexWin;
	}
	
	//barrier( CLK_GLOBAL_MEM_FENCE );
	//work_group_barrier(CLK_LOCAL_MEM_FENCE);

	if (indexWin < numNeur && spikes_Ae_Ai_pre[threadNum] != 0 )  {
		++digits[ assignments[threadNum] ];
		spikes_Ae_Ai_pre[threadNum] = (threadNum == indexWin) ? 1 : 0;
		++spike_count[indexWin];
	}
	
// =================
// =================
// =================
	
	// 2. Update I neurons
	vSyn = 0.0f;
	for (int j = 0; j < numNeur; ++j) {
		vSyn += spikes_Ae_Ai_pos[ threadNum ] ? weightsAeAi[ threadNum * numNeur + j ] : 0.0f ;
	}
	
	//barrier( CLK_GLOBAL_MEM_FENCE );
	
	vI[threadNum] = 0.9048374295234680175781251 * (vI[threadNum] - (-60.0)) + (-60.0);
	
	if( refrac_countI[threadNum] > 0 )
		vSyn = 0.0;
	
	vI[threadNum] += vSyn;

	refrac_countI[threadNum] -= 1;
	
	if ( vI[threadNum] > -40.0 ){	
		refrac_countI[threadNum] =  2;	
		vI[threadNum] =  (-45.0);
		for (int j = 0; j < numNeur; ++j) {
			spikes_Ai_Ae_pre[j] = 0;
		}
		spikes_Ai_Ae_pre[threadNum] = 1;
	}
	
	//barrier( CLK_GLOBAL_MEM_FENCE );
	
	// The arrays are exchanged for the next iteration of time
	
	spikesXePos[threadNum] = spikesXePre[threadNum];
	if( threadNum+numNeur < 784)
		spikesXePos[threadNum+numNeur] = spikesXePre[threadNum+numNeur];
	
	
	spikes_Ae_Ai_pos[threadNum] = spikes_Ae_Ai_pre[threadNum];
	spikes_Ai_Ae_pos[threadNum] = spikes_Ai_Ae_pre[threadNum];
	spikes_Ae_Ai_pre[threadNum] = 0;
	spikes_Ai_Ae_pre[threadNum] = 0;
	
	indexWin = numNeur;
	
	
}
