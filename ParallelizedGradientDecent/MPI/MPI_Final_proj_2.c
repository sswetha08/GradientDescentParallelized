#include <stdio.h>
#include<mpi.h>
#include <math.h>
#include <stdlib.h>
#define a 0.0000002// Set a constant learning rate

float h(int x1,int x2, float b0, float b1,float b2){
    return x2*b2 + x1*b1 + b0;
}

// b0 partial derivative function
float db0(float *x1, float *x2, float *y, int n, float b0, float b1, float b2){
    int i;
    float d = 0;
    for(i=0; i<n; i++){
        d += h(x1[i],x2[i], b0, b1, b2) - y[i];
    }
    return d;
}

// b1 partial derivative function
float db1(float *x1, float *x2, float *y, int n, float b0, float b1, float b2){
    int i;
    float d = 0;
    for(i=0; i<n; i++){
        d += (h(x1[i],x2[i], b0, b1, b2)-y[i])*x1[i];
    }
    return d;
}

// b2 partial derivative function
float db2(float *x1,float *x2, float *y, int n, float b0, float b1,float b2){
    int i;
    float d = 0;
    for(i=0; i<n; i++){
        d += (h(x1[i],x2[i], b0, b1, b2)-y[i])*x2[i];
    }
    return d;
}

float compute_cost(float *x1,float*x2 , float *y, int n,float b0,float b1,float b2)
{
    int i;
    float cost = 0;
    for(i=0; i<n; i++){
        cost += (float)pow(((h(x1[i],x2[i], b0, b1, b2)) - y[i]),2)/(2.0*n);
    }
    return cost;
}
int main(int argc,char* argv[])
{
    int rank,num_procs, i;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
	double t1,t2;
    float b0,b1,b2,learn_b0,learn_b1,temp0=0,temp1=0,j=0,x1_in,x2_in;
    float x1_s[1000],x2_s[1000],y_s[1000],x1_r[1000],x2_r[1000],y_r[1000],grad_b0,grad_b1,grad_b2,grad_b0_r,grad_b1_r,grad_b2_r;
    int n = 500;
    float cost=100,cost_proc;
	float old_cost;	
if(rank==0)
{	
    for(i=0;i<n;i++)
    {
    	x1_s[i] = i;
    	x2_s[i] = pow(i,2);
    	y_s[i] = 2*x1_s[i]+ 3*x2_s[i]+ 1;
    }
    b1=b2=4; b0=1;
  
}
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b0,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b1,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b2,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(x1_s,n/num_procs,MPI_FLOAT,x1_r,n/num_procs,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(x2_s,n/num_procs,MPI_FLOAT,x2_r,n/num_procs,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(y_s,n/num_procs,MPI_FLOAT,y_r,n/num_procs,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	old_cost = 200;
while(/*fabs(old_cost - cost) > 0.00001 ||*/ cost > 0.087)
{
	
	old_cost = cost;
    grad_b0 = db0(x1_r,x2_r,y_r,n/num_procs,b0,b1,b2);
    grad_b1 = db1(x1_r,x2_r,y_r,n/num_procs,b0,b1,b2);
    grad_b2 = db2(x1_r,x2_r,y_r,n/num_procs,b0,b1,b2);
    cost_proc = compute_cost(x1_r,x2_r,y_r,n/num_procs,b0,b1,b2);

    MPI_Reduce(&cost_proc,&cost,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&grad_b0,&grad_b0_r,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&grad_b1,&grad_b1_r,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&grad_b2,&grad_b2_r,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0)
    {
     	b0 = b0 - 6000*a*grad_b0_r/(float)n;
    	b1 = b1 - 90*a*grad_b1_r/(float)n;  
    	b2 = b2 - a*grad_b2_r/(10000*(float)n);
    	//cost = compute_cost(x_s,y_s,n,b0,b1);
    	printf("\nIteration no: %f",j++);
	//printf("\nOld Cost: %f", old_cost);
    	printf("\nCost is : %f", cost);
    	
    	printf("\nb0: %f\tb1: %f\tb2: %f",b0,b1,b2);
    }
    MPI_Bcast(&cost,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b0,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b1,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b2,1,MPI_INT,0,MPI_COMM_WORLD); 
}
	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	if(rank==0)
		{
		printf("\nTime taken for execution : %lf",t2-t1);
		printf("\nEnter the values of x1 and x2 to predict the y value :\n");
		scanf("%f %f",&x1_in,&x2_in);
		printf("The predicted value is : %f \n",h(x1_in,x2_in,b0,b1,b2));
		}
	MPI_Finalize();
    return 0;
}
