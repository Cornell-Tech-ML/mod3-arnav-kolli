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