
__kernel void calc_cost( __global int* X,__global int* Y,__global float* Deriv,__global float* Diff,__global float* Diff_Sqr,float b0,float b1,int size_per_thread)
{
 int id=get_global_id(0);
 int pos = id*size_per_thread;
 int i;
 float hyp;

 for(i=pos;i<pos+size_per_thread;i++)
{
 hyp = b1*X[i] + b0;
 
Diff[i] = hyp - Y[i];
 Deriv[i] = Diff[i]*X[i];
 Diff_Sqr[i] = Diff[i]*Diff[i] ;

/*Diff[i] = 25;
Deriv[i] = 10;
Diff_Sqr[i] = 625; 
*/
}
 
}
