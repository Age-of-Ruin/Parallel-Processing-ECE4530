__kernel void transpose(__global int* a, __global int* aTranspose, int m, int n)
{
    int gidrow = get_global_id(0);
    int gidcol = get_global_id(1);
    
    aTranspose[gidcol*m+gidrow] = a[gidrow*n + gidcol];
}
