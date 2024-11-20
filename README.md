# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

Parallel Check:
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (174)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (174) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        size = int(np.prod(out_shape))---------------------------------------| #0
                                                                             | 
        # Check if shapes and strides match                                  | 
        stride_aligned = (                                                   | 
            len(out_shape) == len(in_shape)                                  | 
            and np.array_equal(out_strides, in_strides)                      | 
            and np.array_equal(out_shape, in_shape)                          | 
        )                                                                    | 
                                                                             | 
        if stride_aligned:                                                   | 
            for i in prange(size):-------------------------------------------| #1
                out[i] = fn(in_storage[i])                                   | 
            return                                                           | 
        else:                                                                | 
            for i in prange(size):-------------------------------------------| #2
                out_index = np.empty(len(out_shape), np.int32)               | 
                in_index = np.empty(len(in_shape), np.int32)                 | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                in_pos = index_to_position(in_index, in_strides)             | 
                out_pos = index_to_position(out_index, out_strides)          | 
                out[out_pos] = fn(in_storage[in_pos])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (197) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (198) is hoisted 
out of the parallel loop labelled #2 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (231)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (231) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        size = int(np.prod(out_shape))-------------------------------------| #3
                                                                           | 
        # Check if all shapes and strides match for direct mapping         | 
        stride_aligned = (                                                 | 
            len(out_shape) == len(a_shape)                                 | 
            and len(out_shape) == len(b_shape)                             | 
            and np.array_equal(out_strides, a_strides)                     | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        )                                                                  | 
                                                                           | 
        if stride_aligned:                                                 | 
            for i in prange(size):-----------------------------------------| #4
                out[i] = fn(a_storage[i], b_storage[i])                    | 
            return                                                         | 
        else:                                                              | 
                                                                           | 
            for i in prange(size):-----------------------------------------| #5
                out_index = np.empty(len(out_shape), np.int32)             | 
                a_index = np.empty(len(a_shape), np.int32)                 | 
                b_index = np.empty(len(b_shape), np.int32)                 | 
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out_pos = index_to_position(out_index, out_strides)        | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #3, #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (261) is hoisted 
out of the parallel loop labelled #5 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (262) is hoisted 
out of the parallel loop labelled #5 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (263) is hoisted 
out of the parallel loop labelled #5 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (296)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (296) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        size = int(np.prod(out_shape))-----------------------------| #6
        reduce_size = a_shape[reduce_dim]                          | 
                                                                   | 
        for i in prange(size):-------------------------------------| #7
            out_index = np.empty(len(out_shape), np.int32)         | 
            in_index = np.empty(len(a_shape), np.int32)            | 
            to_index(i, out_shape, out_index)                      | 
                                                                   | 
            # Copy output index to input index                     | 
            for j in range(len(out_index)):                        | 
                in_index[j] = out_index[j]                         | 
                                                                   | 
            # Initialize with first value                          | 
            out_pos = index_to_position(out_index, out_strides)    | 
                                                                   | 
            # Reduce along dimension                               | 
            for j in range(reduce_size):                           | 
                in_index[reduce_dim] = j                           | 
                pos = index_to_position(in_index, a_strides)       | 
                out[out_pos] = fn(out[out_pos], a_storage[pos])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (309) is hoisted 
out of the parallel loop labelled #7 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (310) is hoisted 
out of the parallel loop labelled #7 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (329)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/arnavkolli/MLE/mod3-arnav-kolli/minitorch/fast_ops.py (329) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    for n in prange(out_shape[0]):----------------------------------------------------------| #8
        for i in range(out_shape[1]):                                                       | 
            for j in range(out_shape[2]):                                                   | 
                sum = 0.0                                                                   | 
                for k in range(a_shape[2]):                                                 | 
                    a_index = n * a_batch_stride + i * a_strides[1] + k * a_strides[2]      | 
                    b_index = n * b_batch_stride + k * b_strides[1] + j * b_strides[2]      | 
                    sum += a_storage[a_index] * b_storage[b_index]                          | 
                out_index = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
                out[out_index] = sum                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


CPU - simple:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  4.447277887247159 correct 39 avg time per epoch: 27.1719s
Epoch  10  loss  1.4312127215600843 correct 49 avg time per epoch: 0.0956s
Epoch  20  loss  0.52788516181617 correct 49 avg time per epoch: 0.0944s
Epoch  30  loss  1.0180985641409372 correct 49 avg time per epoch: 0.0947s
Epoch  40  loss  0.686373853393767 correct 50 avg time per epoch: 0.0949s
Epoch  50  loss  0.7988342514491319 correct 50 avg time per epoch: 0.1151s
Epoch  60  loss  0.19069595845361365 correct 50 avg time per epoch: 0.1942s
Epoch  70  loss  1.0085570534118093 correct 50 avg time per epoch: 0.0956s
Epoch  80  loss  1.1217594624279235 correct 50 avg time per epoch: 0.0958s
Epoch  90  loss  0.9217214393654716 correct 50 avg time per epoch: 0.1011s
Epoch  100  loss  0.5460891324155109 correct 49 avg time per epoch: 0.0956s
Epoch  110  loss  0.05547731887664754 correct 50 avg time per epoch: 0.0935s
Epoch  120  loss  0.010999180695836636 correct 50 avg time per epoch: 0.0949s
Epoch  130  loss  0.6623197186149162 correct 50 avg time per epoch: 0.0940s
Epoch  140  loss  0.5084638302003421 correct 50 avg time per epoch: 0.0943s
Epoch  150  loss  0.2839954770980322 correct 50 avg time per epoch: 0.0934s
Epoch  160  loss  1.1567312301614596 correct 50 avg time per epoch: 0.1062s
Epoch  170  loss  0.8663751332398351 correct 50 avg time per epoch: 0.1596s
Epoch  180  loss  0.7801995322718163 correct 50 avg time per epoch: 0.1492s
Epoch  190  loss  0.22012551522943888 correct 50 avg time per epoch: 0.0962s
Epoch  200  loss  0.1479341284297689 correct 50 avg time per epoch: 0.0954s
Epoch  210  loss  0.00394351106114396 correct 50 avg time per epoch: 0.0956s
Epoch  220  loss  0.7127545015421429 correct 50 avg time per epoch: 0.0946s
Epoch  230  loss  0.975711108071375 correct 50 avg time per epoch: 0.0944s
Epoch  240  loss  0.2753677674190873 correct 50 avg time per epoch: 0.0959s
Epoch  250  loss  0.6200162368336907 correct 50 avg time per epoch: 0.0955s
Epoch  260  loss  0.6230240349646593 correct 50 avg time per epoch: 0.0954s
Epoch  270  loss  0.793603927734221 correct 50 avg time per epoch: 0.0960s
Epoch  280  loss  0.013094169182445272 correct 50 avg time per epoch: 0.0962s
Epoch  290  loss  0.015827636568333858 correct 50 avg time per epoch: 0.1836s
Epoch  300  loss  0.8234216388297604 correct 50 avg time per epoch: 0.1315s
Epoch  310  loss  0.00652886246072058 correct 50 avg time per epoch: 0.0978s
Epoch  320  loss  0.0017748862503583083 correct 50 avg time per epoch: 0.0956s
Epoch  330  loss  0.6289002412720105 correct 50 avg time per epoch: 0.0973s
Epoch  340  loss  0.12208613281086338 correct 50 avg time per epoch: 0.0960s
Epoch  350  loss  0.018732802553885028 correct 50 avg time per epoch: 0.0978s
Epoch  360  loss  0.15913008079159008 correct 50 avg time per epoch: 0.0955s
Epoch  370  loss  0.8501265575113597 correct 50 avg time per epoch: 0.0950s
Epoch  380  loss  0.0052664119730233 correct 50 avg time per epoch: 0.0989s
Epoch  390  loss  0.15075877335347385 correct 50 avg time per epoch: 0.0952s
Epoch  400  loss  0.5985613093928964 correct 50 avg time per epoch: 0.1177s
Epoch  410  loss  0.11135530760798887 correct 50 avg time per epoch: 0.1921s
Epoch  420  loss  0.19620965709602933 correct 50 avg time per epoch: 0.0951s
Epoch  430  loss  0.6763248732446201 correct 50 avg time per epoch: 0.0955s
Epoch  440  loss  0.08602735608842657 correct 50 avg time per epoch: 0.0950s
Epoch  450  loss  0.473061446636093 correct 50 avg time per epoch: 0.0960s
Epoch  460  loss  0.625009701937231 correct 50 avg time per epoch: 0.0955s
Epoch  470  loss  0.004134128041738866 correct 50 avg time per epoch: 0.0944s
Epoch  480  loss  0.6352629827375365 correct 50 avg time per epoch: 0.0944s
Epoch  490  loss  0.0033893843175246867 correct 50 avg time per epoch: 0.0948s

CPU - split:

!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  6.1014478844709235 correct 35 avg time per epoch: 28.0906s
Epoch  10  loss  4.712024354136545 correct 31 avg time per epoch: 0.0965s
Epoch  20  loss  4.5207161548549735 correct 42 avg time per epoch: 0.0962s
Epoch  30  loss  3.9973889683427077 correct 42 avg time per epoch: 0.0952s
Epoch  40  loss  2.3820687998326897 correct 42 avg time per epoch: 0.0949s
Epoch  50  loss  1.5121062882487586 correct 45 avg time per epoch: 0.0968s
Epoch  60  loss  2.1665299850485935 correct 46 avg time per epoch: 0.0956s
Epoch  70  loss  7.106056201845669 correct 45 avg time per epoch: 0.0953s
Epoch  80  loss  2.571884488010909 correct 45 avg time per epoch: 0.0982s
Epoch  90  loss  1.6330624427989102 correct 47 avg time per epoch: 0.0944s
Epoch  100  loss  3.001559403326516 correct 48 avg time per epoch: 0.1073s
Epoch  110  loss  1.3109958656246183 correct 46 avg time per epoch: 0.1914s
Epoch  120  loss  2.1024271357089805 correct 46 avg time per epoch: 0.1037s
Epoch  130  loss  1.602736371595897 correct 50 avg time per epoch: 0.0953s
Epoch  140  loss  0.19364790296094694 correct 49 avg time per epoch: 0.0952s
Epoch  150  loss  1.3102799479425549 correct 50 avg time per epoch: 0.0951s
Epoch  160  loss  1.9800584919165325 correct 50 avg time per epoch: 0.1075s
Epoch  170  loss  1.7352471471791744 correct 50 avg time per epoch: 0.0973s
Epoch  180  loss  0.8302231558916181 correct 49 avg time per epoch: 0.0952s
Epoch  190  loss  0.3926489312706062 correct 48 avg time per epoch: 0.0945s
Epoch  200  loss  1.3763995398517288 correct 49 avg time per epoch: 0.0948s
Epoch  210  loss  0.951362751461471 correct 50 avg time per epoch: 0.0963s
Epoch  220  loss  0.4777670272523314 correct 50 avg time per epoch: 0.1440s
Epoch  230  loss  0.7957738364010588 correct 49 avg time per epoch: 0.1589s
Epoch  240  loss  0.9508574231673705 correct 50 avg time per epoch: 0.0965s
Epoch  250  loss  0.7003141918275394 correct 49 avg time per epoch: 0.0958s
Epoch  260  loss  1.127055768045042 correct 49 avg time per epoch: 0.0942s
Epoch  270  loss  0.901536726753371 correct 49 avg time per epoch: 0.0970s
Epoch  280  loss  0.8933787939011358 correct 49 avg time per epoch: 0.0944s
Epoch  290  loss  0.21122234473822213 correct 50 avg time per epoch: 0.0955s
Epoch  300  loss  1.0385206740110227 correct 50 avg time per epoch: 0.0951s
Epoch  310  loss  0.9349872823704178 correct 49 avg time per epoch: 0.0950s
Epoch  320  loss  0.2650869879948604 correct 50 avg time per epoch: 0.0946s
Epoch  330  loss  0.8417786652309376 correct 50 avg time per epoch: 0.0961s
Epoch  340  loss  0.02852476701232895 correct 49 avg time per epoch: 0.1549s
Epoch  350  loss  0.834681265915013 correct 50 avg time per epoch: 0.1465s
Epoch  360  loss  0.4051758265620973 correct 49 avg time per epoch: 0.0979s
Epoch  370  loss  1.7249545020905759 correct 48 avg time per epoch: 0.0943s
Epoch  380  loss  0.8236654041826212 correct 49 avg time per epoch: 0.0966s
Epoch  390  loss  0.1674918880460339 correct 50 avg time per epoch: 0.1670s
Epoch  400  loss  0.6121229775375984 correct 49 avg time per epoch: 0.1338s
Epoch  410  loss  1.044094944736433 correct 49 avg time per epoch: 0.0927s
Epoch  420  loss  0.8877785490948669 correct 49 avg time per epoch: 0.0942s
Epoch  430  loss  0.5512498607725691 correct 49 avg time per epoch: 0.0927s
Epoch  440  loss  0.04394567666815819 correct 49 avg time per epoch: 0.0933s
Epoch  450  loss  0.7849120125912208 correct 49 avg time per epoch: 0.1671s
Epoch  460  loss  0.07946928334641803 correct 50 avg time per epoch: 0.1346s
Epoch  470  loss  0.5661791384032168 correct 49 avg time per epoch: 0.0939s
Epoch  480  loss  0.8505975888489522 correct 49 avg time per epoch: 0.0977s
Epoch  490  loss  0.8507840407957231 correct 50 avg time per epoch: 0.0936s

CPU - XOR:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  6.485834369986844 correct 34 avg time per epoch: 27.5236s
Epoch  10  loss  4.181635649507905 correct 39 avg time per epoch: 0.0937s
Epoch  20  loss  5.353357285771887 correct 38 avg time per epoch: 0.0930s
Epoch  30  loss  4.621335539065512 correct 45 avg time per epoch: 0.0954s
Epoch  40  loss  4.614175434501773 correct 44 avg time per epoch: 0.1140s
Epoch  50  loss  2.402615414506916 correct 45 avg time per epoch: 0.1697s
Epoch  60  loss  2.280833175427029 correct 45 avg time per epoch: 0.1088s
Epoch  70  loss  2.744223829581904 correct 48 avg time per epoch: 0.0947s
Epoch  80  loss  2.118027328517965 correct 48 avg time per epoch: 0.0953s
Epoch  90  loss  3.15408718439365 correct 49 avg time per epoch: 0.0953s
Epoch  100  loss  2.260651761280821 correct 50 avg time per epoch: 0.0964s
Epoch  110  loss  2.972708170957818 correct 50 avg time per epoch: 0.0967s
Epoch  120  loss  1.3474394017029887 correct 50 avg time per epoch: 0.0954s
Epoch  130  loss  0.719457181645507 correct 50 avg time per epoch: 0.0966s
Epoch  140  loss  1.5769283319818703 correct 50 avg time per epoch: 0.0956s
Epoch  150  loss  1.2529967590675435 correct 50 avg time per epoch: 0.0955s
Epoch  160  loss  1.1200458905370057 correct 50 avg time per epoch: 0.1509s
Epoch  170  loss  0.7780987519898324 correct 50 avg time per epoch: 0.1674s
Epoch  180  loss  1.2236117772095287 correct 49 avg time per epoch: 0.0950s
Epoch  190  loss  0.35248197153331 correct 50 avg time per epoch: 0.0994s
Epoch  200  loss  0.5284288469656645 correct 50 avg time per epoch: 0.0952s
Epoch  210  loss  0.2252219227762711 correct 50 avg time per epoch: 0.0953s
Epoch  220  loss  0.6923217688520176 correct 50 avg time per epoch: 0.0979s
Epoch  230  loss  1.0163642442390044 correct 50 avg time per epoch: 0.0944s
Epoch  240  loss  1.6066823037100335 correct 50 avg time per epoch: 0.0970s
Epoch  250  loss  0.997354814943793 correct 50 avg time per epoch: 0.0966s
Epoch  260  loss  0.6838776419400561 correct 50 avg time per epoch: 0.0953s
Epoch  270  loss  0.5932192064075084 correct 50 avg time per epoch: 0.0938s
Epoch  280  loss  0.4560935110760124 correct 50 avg time per epoch: 0.1504s
Epoch  290  loss  1.0841626346005102 correct 50 avg time per epoch: 0.1535s
Epoch  300  loss  0.3931515848702539 correct 50 avg time per epoch: 0.0948s
Epoch  310  loss  0.2958199924548818 correct 50 avg time per epoch: 0.0944s
Epoch  320  loss  0.040101122655188066 correct 50 avg time per epoch: 0.0975s
Epoch  330  loss  0.6850720497173775 correct 50 avg time per epoch: 0.0945s
Epoch  340  loss  0.5505853354482452 correct 50 avg time per epoch: 0.0929s
Epoch  350  loss  0.9083577788712978 correct 50 avg time per epoch: 0.0954s
Epoch  360  loss  0.34301951467193253 correct 50 avg time per epoch: 0.0963s
Epoch  370  loss  0.07329453217085047 correct 50 avg time per epoch: 0.0939s
Epoch  380  loss  0.31326782587595653 correct 50 avg time per epoch: 0.0942s
Epoch  390  loss  0.18385183130117147 correct 50 avg time per epoch: 0.0950s
Epoch  400  loss  0.2666567129379541 correct 50 avg time per epoch: 0.1472s
Epoch  410  loss  0.25355383757091565 correct 50 avg time per epoch: 0.1530s
Epoch  420  loss  0.381565470305691 correct 50 avg time per epoch: 0.0966s
Epoch  430  loss  0.48614943050435666 correct 50 avg time per epoch: 0.0958s
Epoch  440  loss  0.39385799329725774 correct 50 avg time per epoch: 0.0965s
Epoch  450  loss  0.6882057219831687 correct 50 avg time per epoch: 0.0932s
Epoch  460  loss  0.15588669935335883 correct 50 avg time per epoch: 0.0941s
Epoch  470  loss  0.019137739469759302 correct 50 avg time per epoch: 0.0943s
Epoch  480  loss  0.09538255375802489 correct 50 avg time per epoch: 0.0926s
Epoch  490  loss  0.03982074460882726 correct 50 avg time per epoch: 0.0947s

GPU - simple:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  6.499297434421703 correct 41 avg time per epoch: 3.7063s
Epoch  10  loss  1.9197751767629194 correct 47 avg time per epoch: 1.4180s
Epoch  20  loss  1.5624864412854984 correct 49 avg time per epoch: 1.4396s
Epoch  30  loss  1.0093761781584312 correct 50 avg time per epoch: 1.4690s
Epoch  40  loss  0.17371170289376364 correct 49 avg time per epoch: 1.4861s
Epoch  50  loss  0.2399528378373321 correct 50 avg time per epoch: 1.4406s
Epoch  60  loss  0.048707643051577876 correct 49 avg time per epoch: 1.4363s
Epoch  70  loss  0.46552169838273094 correct 50 avg time per epoch: 1.4414s
Epoch  80  loss  1.356638643074434 correct 50 avg time per epoch: 1.4449s
Epoch  90  loss  1.279434158257644 correct 50 avg time per epoch: 1.5119s
Epoch  100  loss  0.9003941787590035 correct 50 avg time per epoch: 1.4345s
Epoch  110  loss  0.2229860030143969 correct 49 avg time per epoch: 1.4333s
Epoch  120  loss  0.37955170489029577 correct 50 avg time per epoch: 1.4402s
Epoch  130  loss  0.3205340734661026 correct 50 avg time per epoch: 1.4279s
Epoch  140  loss  0.15146754705141452 correct 50 avg time per epoch: 1.4855s
Epoch  150  loss  0.2663391839968446 correct 50 avg time per epoch: 1.4796s
Epoch  160  loss  0.003188194250995965 correct 49 avg time per epoch: 1.4223s
Epoch  170  loss  1.6844471114831703 correct 49 avg time per epoch: 1.4219s
Epoch  180  loss  0.156665770386372 correct 50 avg time per epoch: 1.4192s
Epoch  190  loss  1.3195216080772043 correct 49 avg time per epoch: 1.4268s
Epoch  200  loss  0.21227963971858138 correct 49 avg time per epoch: 1.5105s
Epoch  210  loss  0.4627098326887388 correct 49 avg time per epoch: 1.4247s
Epoch  220  loss  0.0002091716011034311 correct 50 avg time per epoch: 1.4422s
Epoch  230  loss  0.3690672814417795 correct 50 avg time per epoch: 1.4251s
Epoch  240  loss  0.5678558266894985 correct 50 avg time per epoch: 1.4228s
Epoch  250  loss  0.2252912573692845 correct 50 avg time per epoch: 1.4288s
Epoch  260  loss  0.05528824425344531 correct 50 avg time per epoch: 1.4985s
Epoch  270  loss  0.2442328511399805 correct 50 avg time per epoch: 1.4256s
Epoch  280  loss  0.05669256980721413 correct 50 avg time per epoch: 1.5024s
Epoch  290  loss  0.24568613490984567 correct 50 avg time per epoch: 1.4215s
Epoch  300  loss  0.15481859117178567 correct 50 avg time per epoch: 1.4314s
Epoch  310  loss  0.6877523607273394 correct 50 avg time per epoch: 1.5187s
Epoch  320  loss  0.2917983136890633 correct 50 avg time per epoch: 1.4293s
Epoch  330  loss  0.0008120145939394094 correct 49 avg time per epoch: 1.4393s
Epoch  340  loss  0.9885175291344439 correct 49 avg time per epoch: 1.4166s
Epoch  350  loss  0.13481573355785056 correct 50 avg time per epoch: 1.4380s
Epoch  360  loss  0.21235514180511317 correct 50 avg time per epoch: 1.4991s
Epoch  370  loss  0.46431866835711266 correct 50 avg time per epoch: 1.4586s
Epoch  380  loss  0.042166191020348985 correct 50 avg time per epoch: 1.4255s
Epoch  390  loss  0.9661515512722092 correct 49 avg time per epoch: 1.4183s
Epoch  400  loss  0.24521414480816328 correct 50 avg time per epoch: 1.4004s
Epoch  410  loss  0.1263029857864631 correct 50 avg time per epoch: 1.4296s
Epoch  420  loss  0.8805911209786771 correct 49 avg time per epoch: 1.5080s
Epoch  430  loss  0.033871679649384694 correct 50 avg time per epoch: 1.4395s
Epoch  440  loss  0.010313799377997514 correct 50 avg time per epoch: 1.4127s
Epoch  450  loss  0.4402628964953044 correct 50 avg time per epoch: 1.4229s
Epoch  460  loss  0.047630142810975266 correct 49 avg time per epoch: 1.4199s
Epoch  470  loss  0.6226133221602227 correct 50 avg time per epoch: 1.4272s
Epoch  480  loss  0.008300592820716964 correct 50 avg time per epoch: 1.4972s
Epoch  490  loss  0.0014473975268526024 correct 50 avg time per epoch: 1.4142s

GPU - split:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  8.038206334551809 correct 27 avg time per epoch: 5.4499s
Epoch  10  loss  5.817886910783885 correct 40 avg time per epoch: 1.4146s
Epoch  20  loss  2.436481995737376 correct 44 avg time per epoch: 1.4521s
Epoch  30  loss  1.8880525246130573 correct 46 avg time per epoch: 1.4060s
Epoch  40  loss  2.3613897428787576 correct 47 avg time per epoch: 1.4152s
Epoch  50  loss  4.285391746241412 correct 48 avg time per epoch: 1.4755s
Epoch  60  loss  1.9551592471708705 correct 47 avg time per epoch: 1.4285s
Epoch  70  loss  3.824119416030318 correct 48 avg time per epoch: 1.4078s
Epoch  80  loss  1.9029891684396616 correct 48 avg time per epoch: 1.4135s
Epoch  90  loss  1.492759942132136 correct 48 avg time per epoch: 1.4085s
Epoch  100  loss  2.5259572066813045 correct 50 avg time per epoch: 1.4176s
Epoch  110  loss  0.6186687002838005 correct 49 avg time per epoch: 1.4881s
Epoch  120  loss  0.6517605857815482 correct 50 avg time per epoch: 1.4162s
Epoch  130  loss  1.5323057921319996 correct 50 avg time per epoch: 1.4137s
Epoch  140  loss  0.6551958060253857 correct 49 avg time per epoch: 1.4275s
Epoch  150  loss  1.0466460967485824 correct 50 avg time per epoch: 1.4201s
Epoch  160  loss  1.1966711736576567 correct 50 avg time per epoch: 1.4171s
Epoch  170  loss  0.4254163267229776 correct 50 avg time per epoch: 1.4990s
Epoch  180  loss  0.44491252165456774 correct 50 avg time per epoch: 1.4280s
Epoch  190  loss  0.5032167959125051 correct 50 avg time per epoch: 1.5030s
Epoch  200  loss  0.21177260560642464 correct 50 avg time per epoch: 1.4172s
Epoch  210  loss  0.5553883720879792 correct 50 avg time per epoch: 1.4202s
Epoch  220  loss  0.4661019383032515 correct 50 avg time per epoch: 1.4820s
Epoch  230  loss  0.2263988211597723 correct 50 avg time per epoch: 1.4330s
Epoch  240  loss  0.6275223639203881 correct 50 avg time per epoch: 1.4439s
Epoch  250  loss  1.0343555383212792 correct 50 avg time per epoch: 1.4253s
Epoch  260  loss  0.2922670194879926 correct 50 avg time per epoch: 1.4133s
Epoch  270  loss  1.0035739871472016 correct 50 avg time per epoch: 1.4194s
Epoch  280  loss  0.08515670993769038 correct 50 avg time per epoch: 1.4953s
Epoch  290  loss  0.04132650079310202 correct 50 avg time per epoch: 1.4066s
Epoch  300  loss  0.4046001220097235 correct 50 avg time per epoch: 1.4172s
Epoch  310  loss  0.15347445686328157 correct 50 avg time per epoch: 1.4222s
Epoch  320  loss  0.2784300272226537 correct 50 avg time per epoch: 1.4177s
Epoch  330  loss  0.56441863133197 correct 50 avg time per epoch: 1.4316s
Epoch  340  loss  0.25594322708374384 correct 50 avg time per epoch: 1.4776s
Epoch  350  loss  0.15848267505023064 correct 50 avg time per epoch: 1.4189s
Epoch  360  loss  0.1606507430802121 correct 50 avg time per epoch: 1.4069s
Epoch  370  loss  0.024481724128629694 correct 50 avg time per epoch: 1.4175s
Epoch  380  loss  0.09422960402730683 correct 50 avg time per epoch: 1.4106s
Epoch  390  loss  0.3186006779055496 correct 50 avg time per epoch: 1.4391s
Epoch  400  loss  0.10626243828908256 correct 50 avg time per epoch: 1.5563s
Epoch  410  loss  0.07762593084235053 correct 50 avg time per epoch: 1.4090s
Epoch  420  loss  0.043863020655349215 correct 50 avg time per epoch: 1.4124s
Epoch  430  loss  0.05718077553862185 correct 50 avg time per epoch: 1.4007s
Epoch  440  loss  0.06877012158001777 correct 50 avg time per epoch: 1.4167s
Epoch  450  loss  0.1164348992523791 correct 50 avg time per epoch: 1.4153s
Epoch  460  loss  0.4659012255652669 correct 50 avg time per epoch: 1.4831s
Epoch  470  loss  0.05905442718152658 correct 50 avg time per epoch: 1.4062s
Epoch  480  loss  0.2565233306180501 correct 50 avg time per epoch: 1.4050s
Epoch  490  loss  0.031106020583303917 correct 50 avg time per epoch: 1.4114s

GPU - xor:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  7.6749187158654975 correct 32 avg time per epoch: 5.5468s
Epoch  10  loss  4.512498112912729 correct 46 avg time per epoch: 1.4319s
Epoch  20  loss  3.4301444095866436 correct 45 avg time per epoch: 1.4166s
Epoch  30  loss  2.3534803614202966 correct 46 avg time per epoch: 1.4109s
Epoch  40  loss  1.7294783542242615 correct 46 avg time per epoch: 1.4169s
Epoch  50  loss  2.4145569637739297 correct 45 avg time per epoch: 1.4776s
Epoch  60  loss  3.3995059910833296 correct 46 avg time per epoch: 1.4124s
Epoch  70  loss  3.2321060716876677 correct 48 avg time per epoch: 1.4121s
Epoch  80  loss  3.6359345085442607 correct 47 avg time per epoch: 1.4167s
Epoch  90  loss  1.5261550348477826 correct 48 avg time per epoch: 1.4214s
Epoch  100  loss  2.0308690827074996 correct 47 avg time per epoch: 1.4319s
Epoch  110  loss  1.3776892634290199 correct 48 avg time per epoch: 1.4739s
Epoch  120  loss  1.2199709901305036 correct 48 avg time per epoch: 1.4195s
Epoch  130  loss  0.7994568480456482 correct 48 avg time per epoch: 1.4209s
Epoch  140  loss  1.448291054018209 correct 49 avg time per epoch: 1.4155s
Epoch  150  loss  3.2400668108414274 correct 48 avg time per epoch: 1.4234s
Epoch  160  loss  0.8127486640617767 correct 49 avg time per epoch: 1.4528s
Epoch  170  loss  1.1813803337904585 correct 49 avg time per epoch: 1.4644s
Epoch  180  loss  1.13825100414587 correct 49 avg time per epoch: 1.4281s
Epoch  190  loss  1.9942183249105752 correct 49 avg time per epoch: 1.4204s
Epoch  200  loss  0.5550931705766708 correct 49 avg time per epoch: 1.4353s
Epoch  210  loss  1.2833035196376041 correct 49 avg time per epoch: 1.5036s
Epoch  220  loss  1.5093879917087685 correct 50 avg time per epoch: 1.4984s
Epoch  230  loss  0.0683477653216786 correct 49 avg time per epoch: 1.4206s
Epoch  240  loss  0.5533798980557253 correct 49 avg time per epoch: 1.4152s
Epoch  250  loss  0.21551201838374495 correct 49 avg time per epoch: 1.4107s
Epoch  260  loss  0.8606941459173961 correct 50 avg time per epoch: 1.4170s
Epoch  270  loss  0.07598300524746225 correct 50 avg time per epoch: 1.4168s
Epoch  280  loss  0.968551383092292 correct 49 avg time per epoch: 1.5045s
Epoch  290  loss  0.5882359809095268 correct 49 avg time per epoch: 1.4308s
Epoch  300  loss  2.1103069238472543 correct 50 avg time per epoch: 1.4360s
Epoch  310  loss  0.3141348900091503 correct 49 avg time per epoch: 1.4318s
Epoch  320  loss  0.295513795572391 correct 50 avg time per epoch: 1.4264s
Epoch  330  loss  1.097252766876292 correct 49 avg time per epoch: 1.4611s
Epoch  340  loss  0.03444538980779395 correct 50 avg time per epoch: 1.4636s
Epoch  350  loss  1.0619811763337283 correct 49 avg time per epoch: 1.4190s
Epoch  360  loss  1.4197573601839557 correct 50 avg time per epoch: 1.4205s
Epoch  370  loss  0.22463270731142776 correct 49 avg time per epoch: 1.4329s
Epoch  380  loss  1.0958346567224369 correct 50 avg time per epoch: 1.4169s
Epoch  390  loss  0.30286164428352474 correct 50 avg time per epoch: 1.4824s
Epoch  400  loss  0.2913668077816931 correct 50 avg time per epoch: 1.4135s
Epoch  410  loss  0.7385591525404334 correct 49 avg time per epoch: 1.4085s
Epoch  420  loss  0.06693142143240007 correct 50 avg time per epoch: 1.4868s
Epoch  430  loss  0.8132010683760205 correct 50 avg time per epoch: 1.4055s
Epoch  440  loss  0.45537188120229577 correct 50 avg time per epoch: 1.4355s
Epoch  450  loss  0.018098885888765612 correct 50 avg time per epoch: 1.4448s
Epoch  460  loss  0.5918207398680141 correct 50 avg time per epoch: 1.3981s
Epoch  470  loss  0.9868895119031131 correct 50 avg time per epoch: 1.4140s
Epoch  480  loss  0.41374694124906164 correct 50 avg time per epoch: 1.4014s
Epoch  490  loss  0.4854752055893098 correct 50 avg time per epoch: 1.4033s

Bigger model:
!cd $DIR; PYTHONPATH=/content/$DIR python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple --RATE 0.05

Epoch  0  loss  64.77253473197128 correct 29 avg time per epoch: 32.8912s
Epoch  10  loss  17.308770784665267 correct 49 avg time per epoch: 1.0816s
Epoch  20  loss  1.7151514603473148 correct 46 avg time per epoch: 1.0709s
Epoch  30  loss  1.1160376470894833 correct 46 avg time per epoch: 1.0748s
Epoch  40  loss  0.24479064533503384 correct 50 avg time per epoch: 1.0815s
Epoch  50  loss  0.331066986222543 correct 50 avg time per epoch: 1.0746s
Epoch  60  loss  0.032592703178170854 correct 50 avg time per epoch: 0.9470s
Epoch  70  loss  0.4238492456625239 correct 48 avg time per epoch: 1.0785s
Epoch  80  loss  0.3918492486552528 correct 50 avg time per epoch: 1.0673s
Epoch  90  loss  0.11569340050507898 correct 50 avg time per epoch: 1.0739s
Epoch  100  loss  0.38544566876264374 correct 50 avg time per epoch: 1.0786s
Epoch  110  loss  0.2388674269271922 correct 50 avg time per epoch: 1.0740s
Epoch  120  loss  0.1559714683178988 correct 50 avg time per epoch: 0.9965s
Epoch  130  loss  0.2583274392912329 correct 50 avg time per epoch: 1.0243s
Epoch  140  loss  0.018447683762220213 correct 50 avg time per epoch: 1.0710s
Epoch  150  loss  0.06415180078423914 correct 50 avg time per epoch: 1.0720s
Epoch  160  loss  0.04152010891357506 correct 50 avg time per epoch: 1.0717s
Epoch  170  loss  0.03369600401583581 correct 50 avg time per epoch: 1.0718s
Epoch  180  loss  0.06584154336162097 correct 50 avg time per epoch: 1.1769s
Epoch  190  loss  0.00902226073174123 correct 50 avg time per epoch: 0.9778s
Epoch  200  loss  0.040546306205375346 correct 50 avg time per epoch: 1.0700s
Epoch  210  loss  0.07884897916094927 correct 50 avg time per epoch: 1.0768s
Epoch  220  loss  0.21175812620110276 correct 50 avg time per epoch: 1.0972s
Epoch  230  loss  0.004649552089920609 correct 50 avg time per epoch: 1.1051s
Epoch  240  loss  0.02347696161345904 correct 50 avg time per epoch: 1.0863s
Epoch  250  loss  0.004627881737073138 correct 50 avg time per epoch: 1.0178s
Epoch  260  loss  0.00979733941002084 correct 50 avg time per epoch: 1.0370s
Epoch  270  loss  0.15058162811297002 correct 50 avg time per epoch: 1.0887s
Epoch  280  loss  0.08135674089330852 correct 50 avg time per epoch: 1.0803s
Epoch  290  loss  0.05626873751813126 correct 50 avg time per epoch: 1.0806s
Epoch  300  loss  0.012596678548086757 correct 50 avg time per epoch: 1.0950s
Epoch  310  loss  0.0017365073804417627 correct 50 avg time per epoch: 1.0752s
Epoch  320  loss  0.09038960705187718 correct 50 avg time per epoch: 0.9646s
Epoch  330  loss  0.133598296398475 correct 50 avg time per epoch: 1.0772s
Epoch  340  loss  0.011845953144378899 correct 50 avg time per epoch: 1.0776s
Epoch  350  loss  0.03486279507669028 correct 50 avg time per epoch: 1.0844s
Epoch  360  loss  0.1191885414891384 correct 50 avg time per epoch: 1.0840s
Epoch  370  loss  0.026675877197327467 correct 50 avg time per epoch: 1.0845s
Epoch  380  loss  0.07188967302430443 correct 50 avg time per epoch: 1.1366s
Epoch  390  loss  0.0883408767122253 correct 50 avg time per epoch: 1.0375s
Epoch  400  loss  0.05254117500808704 correct 50 avg time per epoch: 1.0817s
Epoch  410  loss  0.08307575539606644 correct 50 avg time per epoch: 1.0813s
Epoch  420  loss  0.04961684533081092 correct 50 avg time per epoch: 1.0739s
Epoch  430  loss  0.03531039198034466 correct 50 avg time per epoch: 1.0847s
Epoch  440  loss  0.00462010736717663 correct 50 avg time per epoch: 1.0812s
Epoch  450  loss  0.02269844493075793 correct 50 avg time per epoch: 0.9584s
Epoch  460  loss  0.000652764320496205 correct 50 avg time per epoch: 1.0818s
Epoch  470  loss  0.005944250933352236 correct 50 avg time per epoch: 1.0901s
Epoch  480  loss  0.02515920819099737 correct 50 avg time per epoch: 1.0837s
Epoch  490  loss  0.0016577725244279837 correct 50 avg time per epoch: 1.0895s