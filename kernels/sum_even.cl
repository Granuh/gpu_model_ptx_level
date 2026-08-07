kernel void sum_even(global int *glob, global int *output, local int *loc, int TS) {
    int my_elem = get_local_id(0); // MAP to SMs each of WG size
    int my_unit = get_global_id(0) / get_local_size(0); //0..SM
    int sum = 0;
    for (int i = 0; i < TS; i++) {
	int idx = i + get_global_id(0) * TS; 
    if (glob[idx] % 2 == 0)
	sum += glob[idx]; 
    }
    barrier (CLK_LOCAL_MEM_FENCE); // REDUCE local
    loc[my_elem] = sum;
    if (my_elem == 0) {
	for (int i = 1; i < get_local_size(0); i++) { 
	    atomic_add(loc, loc[i]);
	}
	output[my_unit] = loc[0];
    } 
}