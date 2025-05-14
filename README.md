# SolvingJigsawPuzzles


# REMAINING TASKS:
- Build function to import various models ✅
- Bug fix training ✅
- Move the preprocessing to the dataloader ✅
- Set up inference phase ✅
- bug in err/debug_5490.err ✅
- bug in err/debug_5488.err orerr/debug_5489.err ✅
- freeze encoder or heads weights ✅
- modify model to do only one encoder pass ❌
- parallelize training ❌
- logger during training: mlflow.start_run ✅
- better models saving layout ✅
- save some images for the coloring accuracy ✅
- Add argparse/yaml ✅
- IMPORTANT TO SOLVE: bug in loading the model pretrained because of self.head ✅
- implement the choice of jigsaw predicitons ✅
- use images of larger size ✅
- change jigsaw patch size -> 3 by 3 patches ✅
- try coloring/jigsaw with data augmentation ✅

HIGH:
- function to remove gray scale images
- try coloring/jigsaw with data augmentation
- download coco and put in the right format ✅
- use images of larger size 
- change jigsaw patch size -> 3 by 3 patches
MEDIUM:
- do the sequential mode for the model and add method 'reconstruction' or 'sequential' to switch modes 
- new pixel shuffle with 2 steps
- new jigsaw head
- write reconstruction function to rebuild jigsaw images
OPTIONAL :
-  create the noise head.