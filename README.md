
## Closed-loop Controller

chmod a+x file.sh


## Important Things:

1. When you change my files, especially tabular files (mapping table, Wound_1.csv and etc), make sure you do it using Excel 
   and make sure you don't change the header!!!
   Otherwise it will destroy the file format, and we will have wrong data stored!!!


## Operating Procedure:
1. Double Click "DARPA-BETR-UCSC-Closed-Loop-Controller.sh"
2. In the terminal popped up, 
   1. Type in wound number; For Example, 2 and then hit enter.
   2. Type in reference channels (channels setting current to zero), slip each channel by ",". 
      For example, if you want channel 1,2,3 to be zero, type in 1, 2, 3 and then hit enter;
   3. Type in the experiment duration. For example, if you want to run for 12 hours, type in 12 and then hit enter
3. The closed-loop will start actuating when there are new device images comes in, 
   and shut off either by physicians GUI or we have delivered enough drug or we have exceed time limit.

## TODOs:
1. Drug concentration should be calculated and reset daily
2. There should be two different sets of min/max current for EF, flx separately