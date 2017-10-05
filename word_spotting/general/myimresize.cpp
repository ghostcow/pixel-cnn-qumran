#include "mex.h"

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of input and output arguments */    
//     printf("-I00-%d %d\n",nlhs,nrhs);
    if (nrhs != 3) { 
        mexErrMsgIdAndTxt( "MATLAB:myimresize:minrhs",
                "2 input arguments required.");
    } 
    if(nlhs > 1){
        mexErrMsgIdAndTxt( "MATLAB:myimresize:maxlhs",
                "Too many output arguments.");
    }
    

    double *A = mxGetPr(prhs[0]);
    size_t h = (size_t)mxGetScalar(prhs[1]);
    size_t w = (size_t)mxGetScalar(prhs[2]);
    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);
//     printf("-I0-%d %d\n",m,n);
    mxArray *pB = mxCreateNumericMatrix(h,n,mxSINGLE_CLASS,mxREAL);
    mxArray *pD = mxCreateNumericMatrix(h,w,mxSINGLE_CLASS,mxREAL);
    plhs[0] = pD;
    float *B = (float *)mxGetData(pB);
    float *D = (float *)mxGetData(pD);
    float r = (float)h/float(m);
    float p=1.0f,t,b;
    size_t s, s2;
    //printf("-I1-%d %d\n",m,n);
    if (r<=1) {
         //printf("-I2a-%d %d %f\n",m,n,r);
        s=m*n;
        for (size_t i=0, j=0; i<s; i++) {
            if (p<r) {
                t=r-p;
                B[j++]+=p*A[i];
                B[j]+=t*A[i];
                p=1-t;
            } else {
                B[j]+=r*A[i];
                p-=r;
            }            
        }
    } else {
//          printf("-I2b-%d %d\n",m,n);
        s=h*n;
        p=r;
        for (size_t i=0, j=0; i<s; i++) {
            if (p<1) {
                t=1-p;
                B[i]+=p*A[j];
                j++;
                B[i]+=t*A[j];
                p=r-t;
            } else {
                B[i]+=A[j];
                p-=1;
            }            
        }  
    }
//     printf("-I3-%d %d\n",m,n);
    r = (float)w/float(n);    
    if (r<=1) {
        s=h*n;
        s2=h*w;
        for (size_t c=0; c<h; c++) {
            p=1.0f;
            for (size_t i=c, j=c; i<s; i+=h) {
                b = B[i];
                if (p<r) {
                    t=r-p;
                    D[j]+=p*b;
                    j+=h;
                    if (j<s2) D[j]+=t*b;
                    p=1-t;
                } else {
                    D[j]+=r*b;
                    p-=r;
                }
            }
        }
    } else {
//         printf("-I4b-%d %d\n",m,n);
        s=h*w;
        s2=h*n;
        for (size_t c=0; c<h; c++) {
            p=r;
            for (size_t i=c, j=c; i<s; i+=h) {
                b = B[j];
                if (p<1) {
                    t=1-p;
                    D[i]+=p*B[j];
                    j+=h;
                    if (j<s2) D[i]+=t*B[j];
                    p=r-t;
                } else {
                    D[i]+=r*b;
                    p-=1;
                }
            }
        }
    }
//     printf("-I5-%d %d\n",m,n);
    mxDestroyArray(pB);   
    return;
}