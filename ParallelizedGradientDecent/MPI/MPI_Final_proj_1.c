#include <stdio.h>
#include<mpi.h>
#include <math.h>
#include <stdlib.h>
#define a 0.0000205 // Set a constant learning rate

float h(int x, float b0, float b1){
    return x*b1+b0;
}

// b0 partial derivative function
float db0(float *x, float *y, int n, float b0, float b1){
    int i;
    float d = 0;
    for(i=0; i<n; i++){
        d += h(x[i], b0, b1) - y[i];
    }
    return d;
}

// b1 partial derivative function
float db1(float *x, float *y, int n, float b0, float b1){
    int i;
    float d = 0;
    for(i=0; i<n; i++){
        d += (h(x[i], b0, b1) - y[i])*x[i];
    }
    return d;
}

float compute_cost(float *x, float *y, int n,float b0,float b1)
{
    int i;
    float cost = 0;
    for(i=0; i<n; i++){
        cost += (float)pow((h(x[i], b0, b1) - y[i]),2)/2.0*n;
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
    float b0,b1,learn_b0,learn_b1,temp0=0,temp1=0,j=0;
    float x_s[1000],y_s[1000],x_r[1000],y_r[1000],grad_b0,grad_b1,grad_b0_r,grad_b1_r,x_in;
    int n = 500;
    float cost=100,cost_proc;
	float old_cost;	
if(rank==0)
{	
	printf("The equation is of the form b0 + b1 * X = Y\n");
    for(i=0;i<n;i++)
    {
    	x_s[i] = i;
    	y_s[i] = i*2 + 4;
    }
    b0=b1=0;
    printf("Imported training dataset X and Y \n");
    printf("Hit enter to predict b0 and b1 using the training dataset\n");
    getchar();
  
}
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b0,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b1,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(x_s,n/num_procs,MPI_FLOAT,x_r,n/num_procs,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(y_s,n/num_procs,MPI_FLOAT,y_r,n/num_procs,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();
	old_cost = 200;
while(fabs(old_cost - cost) > 0.00001 || cost > 0.5)
{
	
	old_cost = cost;
    grad_b0 = db0(x_r,y_r,n/num_procs,b0,b1);
    grad_b1 = db1(x_r,y_r,n/num_procs,b0,b1);
    cost_proc = compute_cost(x_r,y_r,n/num_procs,b0,b1);

    MPI_Reduce(&cost_proc,&cost,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&grad_b0,&grad_b0_r,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&grad_b1,&grad_b1_r,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0)
    {
     	b0 = b0 - 50*a*grad_b0_r/(float)n;
    	b1 = b1 - a*grad_b1_r/(float)n;  
    	//cost = compute_cost(x_s,y_s,n,b0,b1);
    	printf("\nIteration no: %f",j++);
		//printf("\nOld Cost: %f", old_cost);
    	printf("\nCost is : %f", cost);
    	
    	printf("\nb0: %f\tb1: %f",b0,b1);
    }
    MPI_Bcast(&cost,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b0,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&b1,1,MPI_INT,0,MPI_COMM_WORLD); 
}

	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	if(rank==0)
	{
		printf("\nTime taken : %lf\n",t2-t1);
		printf("Model sucessfully trained on the dataset\n");
		printf("\nEnter the value of X to predict the Y value :\n");
		scanf("%f",&x_in);
		printf("The predicted value is : %f \n",h(x_in,b0,b1));
		}
	MPI_Finalize();
    return 0;
}
