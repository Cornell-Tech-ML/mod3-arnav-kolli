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

CPU - simple:


CPU - split:

arnavkolli@MacBook-Pro-2 mod3-arnav-kolli % python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  7.179654556437834 correct 31 avg time per epoch: 11.1828s
Epoch  10  loss  4.275303629243521 correct 36 avg time per epoch: 0.1548s
Epoch  20  loss  3.7283112246151493 correct 43 avg time per epoch: 0.1538s
Epoch  30  loss  5.2361936914048 correct 38 avg time per epoch: 0.1540s
Epoch  40  loss  2.8360114918298303 correct 46 avg time per epoch: 0.1524s
Epoch  50  loss  2.9910238777109215 correct 46 avg time per epoch: 0.1528s
Epoch  60  loss  4.187950274939677 correct 48 avg time per epoch: 0.1534s
Epoch  70  loss  2.3968868278290643 correct 48 avg time per epoch: 0.1538s
Epoch  80  loss  2.217168259269237 correct 48 avg time per epoch: 0.1574s
Epoch  90  loss  3.379551450632878 correct 49 avg time per epoch: 0.1523s
Epoch  100  loss  1.3115396279045692 correct 47 avg time per epoch: 0.1525s
Epoch  110  loss  1.373743541881255 correct 46 avg time per epoch: 0.1546s
Epoch  120  loss  1.506995009959202 correct 49 avg time per epoch: 0.1540s
Epoch  130  loss  1.0139670243692953 correct 49 avg time per epoch: 0.1523s
Epoch  140  loss  2.6955316328228935 correct 49 avg time per epoch: 0.1529s
Epoch  150  loss  1.3332564179348587 correct 47 avg time per epoch: 0.1541s
Epoch  160  loss  0.8508471410015745 correct 50 avg time per epoch: 0.1540s
Epoch  170  loss  0.8045944259639658 correct 49 avg time per epoch: 0.1540s
Epoch  180  loss  1.3465855558238353 correct 50 avg time per epoch: 0.1521s
Epoch  190  loss  0.9186402007310072 correct 49 avg time per epoch: 0.1525s
Epoch  200  loss  0.8279106191315552 correct 48 avg time per epoch: 0.1537s
Epoch  210  loss  0.7902194039182011 correct 50 avg time per epoch: 0.1547s
Epoch  220  loss  0.46312177769217694 correct 49 avg time per epoch: 0.1521s
Epoch  230  loss  0.3052660588373749 correct 50 avg time per epoch: 0.1524s
Epoch  240  loss  0.6391967798111163 correct 50 avg time per epoch: 0.1537s
Epoch  250  loss  1.4891274567036952 correct 49 avg time per epoch: 0.1539s
Epoch  260  loss  0.9247503504846113 correct 50 avg time per epoch: 0.1530s
Epoch  270  loss  0.3304439679444488 correct 50 avg time per epoch: 0.1522s
Epoch  280  loss  1.6075570418471714 correct 50 avg time per epoch: 0.1521s
Epoch  290  loss  1.3837534246449237 correct 49 avg time per epoch: 0.1535s
Epoch  300  loss  1.058218068822465 correct 50 avg time per epoch: 0.1534s
Epoch  310  loss  0.4261457663787623 correct 50 avg time per epoch: 0.1531s
Epoch  320  loss  1.1709477665878785 correct 50 avg time per epoch: 0.1516s
Epoch  330  loss  0.7841714994693155 correct 49 avg time per epoch: 0.1534s
Epoch  340  loss  0.8572209082885585 correct 50 avg time per epoch: 0.1542s
Epoch  350  loss  1.6950586421129075 correct 50 avg time per epoch: 0.1530s
Epoch  360  loss  0.6101202064457776 correct 50 avg time per epoch: 0.1559s
Epoch  370  loss  0.8564795770585625 correct 50 avg time per epoch: 0.1528s
Epoch  380  loss  0.008875726951272747 correct 50 avg time per epoch: 0.1548s
Epoch  390  loss  1.3462503325477437 correct 49 avg time per epoch: 0.1538s
Epoch  400  loss  0.7184321776726855 correct 50 avg time per epoch: 0.1527s
Epoch  410  loss  0.7024278099163334 correct 50 avg time per epoch: 0.1522s
Epoch  420  loss  0.5161526601089521 correct 50 avg time per epoch: 0.1533s
Epoch  430  loss  0.17947743061638446 correct 50 avg time per epoch: 0.1537s
Epoch  440  loss  0.3600135968241168 correct 50 avg time per epoch: 0.1537s
Epoch  450  loss  0.12487385959633078 correct 50 avg time per epoch: 0.1514s
Epoch  460  loss  0.6365211270157528 correct 50 avg time per epoch: 0.1522s
Epoch  470  loss  0.3245253950522289 correct 50 avg time per epoch: 0.1546s
Epoch  480  loss  0.41374526740882084 correct 50 avg time per epoch: 0.1542s
Epoch  490  loss  0.2687669825578364 correct 50 avg time per epoch: 0.1524s

CPU - XOR:
(.venv) arnavkolli@MacBook-Pro-2 mod3-arnav-kolli % python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  7.673359645751777 correct 24 avg time per epoch: 11.1253s
Epoch  10  loss  4.728496003242975 correct 40 avg time per epoch: 0.1549s
Epoch  20  loss  3.8736039200789913 correct 41 avg time per epoch: 0.1532s
Epoch  30  loss  5.3078190143892625 correct 44 avg time per epoch: 0.1518s
Epoch  40  loss  4.348707642935453 correct 44 avg time per epoch: 0.1527s
Epoch  50  loss  3.8120746721132086 correct 45 avg time per epoch: 0.1532s
Epoch  60  loss  3.534975242305894 correct 46 avg time per epoch: 0.1529s
Epoch  70  loss  3.478664778407494 correct 46 avg time per epoch: 0.1521s
Epoch  80  loss  4.604469775826845 correct 46 avg time per epoch: 0.1518s
Epoch  90  loss  2.86079216724333 correct 46 avg time per epoch: 0.1533s
Epoch  100  loss  1.847114995617879 correct 47 avg time per epoch: 0.1529s
Epoch  110  loss  1.4743681538258866 correct 47 avg time per epoch: 0.1526s
Epoch  120  loss  2.6372500847879325 correct 47 avg time per epoch: 0.1556s
Epoch  130  loss  0.8208141629393111 correct 49 avg time per epoch: 0.1515s
Epoch  140  loss  1.8887350817470068 correct 49 avg time per epoch: 0.1541s
Epoch  150  loss  1.2351372878385587 correct 49 avg time per epoch: 0.1534s
Epoch  160  loss  1.532424364430582 correct 49 avg time per epoch: 0.1521s
Epoch  170  loss  1.6643301138610247 correct 50 avg time per epoch: 0.1518s
Epoch  180  loss  0.8637194698296359 correct 49 avg time per epoch: 0.1529s
Epoch  190  loss  2.575234647832585 correct 48 avg time per epoch: 0.1528s
Epoch  200  loss  0.9050325807828768 correct 50 avg time per epoch: 0.1527s
Epoch  210  loss  1.6285642935120328 correct 47 avg time per epoch: 0.1524s
Epoch  220  loss  1.4929334941456365 correct 49 avg time per epoch: 0.1654s
Epoch  230  loss  0.9206361507751649 correct 50 avg time per epoch: 0.1529s
Epoch  240  loss  0.1931460123318462 correct 50 avg time per epoch: 0.1547s
Epoch  250  loss  1.2782726430197633 correct 49 avg time per epoch: 0.1518s
Epoch  260  loss  0.8467397106477196 correct 48 avg time per epoch: 0.1522s
Epoch  270  loss  1.383495615731046 correct 47 avg time per epoch: 0.1607s
Epoch  280  loss  0.7671019018720744 correct 50 avg time per epoch: 0.1536s
Epoch  290  loss  2.3405161754590544 correct 48 avg time per epoch: 0.1525s
Epoch  300  loss  0.7320852504927224 correct 49 avg time per epoch: 0.1513s
Epoch  310  loss  0.490812949457041 correct 50 avg time per epoch: 0.1522s
Epoch  320  loss  0.40233255332328777 correct 50 avg time per epoch: 0.1533s
Epoch  330  loss  0.1182934732601559 correct 50 avg time per epoch: 0.1531s
Epoch  340  loss  0.6331467533999035 correct 50 avg time per epoch: 0.1515s
Epoch  350  loss  1.293979517328434 correct 50 avg time per epoch: 0.1517s
Epoch  360  loss  0.7699591428223453 correct 50 avg time per epoch: 0.1519s
Epoch  370  loss  0.7629177439902688 correct 50 avg time per epoch: 0.1532s
Epoch  380  loss  1.652706289179588 correct 48 avg time per epoch: 0.1525s
Epoch  390  loss  0.5293252466773453 correct 50 avg time per epoch: 0.1515s
Epoch  400  loss  0.007107275848623615 correct 50 avg time per epoch: 0.1528s
Epoch  410  loss  0.737547682710533 correct 49 avg time per epoch: 0.1525s
Epoch  420  loss  0.3248896010221403 correct 50 avg time per epoch: 0.1534s
Epoch  430  loss  0.31805142620181887 correct 50 avg time per epoch: 0.1525s
Epoch  440  loss  0.4634901538510357 correct 50 avg time per epoch: 0.1550s
Epoch  450  loss  0.0777063141553308 correct 50 avg time per epoch: 0.1525s
Epoch  460  loss  0.5660108098418828 correct 50 avg time per epoch: 0.1530s
Epoch  470  loss  0.03126854901846196 correct 50 avg time per epoch: 0.1530s
Epoch  480  loss  0.08006564712903746 correct 50 avg time per epoch: 0.1516s
Epoch  490  loss  0.1356835050440753 correct 50 avg time per epoch: 0.1519s

